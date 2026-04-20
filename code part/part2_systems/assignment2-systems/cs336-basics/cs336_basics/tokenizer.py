from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache

import regex as re
"""
encode数据流:
text->split_special_tokens->ordinary text pretokens->each pretoken to bytes->apply BPE merges->map bytes token to id
decode数据流:
ids->vocab lookup to bytes->concatenate bytes->utf-8 decode
"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZER_RE = re.compile(PAT)


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 printable remapping for byte values."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]

    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    return dict(zip(bs, [chr(codepoint) for codepoint in cs]))


def normalize_special_tokens(special_tokens: Sequence[str] | None) -> list[str]:
    """Deduplicate special tokens and prefer longer matches first."""
    if not special_tokens:
        return []
    return sorted(set(special_tokens), key=lambda token: (-len(token), token))


def build_special_token_regex(special_tokens: Sequence[str] | None):
    normalized_special_tokens = normalize_special_tokens(special_tokens)
    if not normalized_special_tokens:
        return None

    pattern = "(" + "|".join(re.escape(token) for token in normalized_special_tokens) + ")"
    return re.compile(pattern)


def split_special_tokens(text: str, special_tokens: Sequence[str] | None) -> list[tuple[bool, str]]:
    """
    Split text into ordinary and special-token segments.

    Returns pairs of (is_special, segment).
    """
    if not text:
        return []

    normalized_special_tokens = normalize_special_tokens(special_tokens)
    if not normalized_special_tokens:
        return [(False, text)]

    special_split_re = build_special_token_regex(normalized_special_tokens)
    assert special_split_re is not None

    special_token_set = set(normalized_special_tokens)
    segments: list[tuple[bool, str]] = []

    for part in special_split_re.split(text):
        if not part:
            continue
        segments.append((part in special_token_set, part))

    return segments


def pretokenize_text(text: str) -> Iterator[str]:
    """Yield GPT-2 style pre-tokens from ordinary text."""
    for match in PRETOKENIZER_RE.finditer(text):
        yield match.group(0)


def iter_text_units(text: str, special_tokens: Sequence[str] | None) -> Iterator[tuple[bool, str]]:
    """
    Yield text units after special-token handling.

    Special tokens are yielded as (True, token).
    Ordinary pre-tokens are yielded as (False, pretoken).
    """
    for is_special, segment in split_special_tokens(text, special_tokens):
        if is_special:
            yield True, segment
        else:
            for pretoken in pretokenize_text(segment):
                yield False, pretoken


def bytes_to_byte_tokens(data: bytes) -> list[bytes]:
    """Convert a bytes object into single-byte tokens."""
    return [bytes([byte]) for byte in data]


def pretoken_to_byte_tokens(pretoken: str) -> list[bytes]:
    """Convert a pre-token string into single-byte UTF-8 byte tokens."""
    return bytes_to_byte_tokens(pretoken.encode("utf-8"))


def get_adjacent_pairs(parts: Sequence[bytes]) -> list[tuple[bytes, bytes]]:
    """Return all adjacent token pairs in order."""
    return [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]


def merge_pair_in_sequence(parts: Sequence[bytes], pair_to_merge: tuple[bytes, bytes]) -> list[bytes]:
    """Merge every non-overlapping occurrence of a pair in one left-to-right pass."""
    merged: list[bytes] = []
    i = 0

    while i < len(parts):
        if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair_to_merge:
            merged.append(parts[i] + parts[i + 1])
            i += 2
        else:
            merged.append(parts[i])
            i += 1

    return merged


def merge_pretoken(pretoken: str, merge_ranks: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    """
    Apply BPE merges to a single pre-token.

    We repeatedly merge the adjacent pair with the smallest rank until no
    adjacent pair is mergeable.
    """
    parts = pretoken_to_byte_tokens(pretoken)

    while len(parts) >= 2:
        best_pair: tuple[bytes, bytes] | None = None
        best_rank: int | None = None

        for pair in get_adjacent_pairs(parts):
            rank = merge_ranks.get(pair)
            if rank is None:
                continue

            if best_rank is None or rank < best_rank:
                best_pair = pair
                best_rank = rank

        if best_pair is None:
            break

        parts = merge_pair_in_sequence(parts, best_pair)

    return parts


def _encode_pretokens_to_ids(
    pretokens: Iterable[str],
    token_to_id: dict[bytes, int],
    merge_ranks: dict[tuple[bytes, bytes], int],
) -> Iterator[int]:
    for pretoken in pretokens:
        for part in merge_pretoken(pretoken, merge_ranks):
            yield token_to_id[part]


def _collect_pretoken_counts(text: str, special_tokens: Sequence[str] | None) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()

    for is_special, unit in iter_text_units(text, special_tokens):
        if is_special:
            continue

        token = tuple(pretoken_to_byte_tokens(unit))
        if token:
            counts[token] += 1

    return counts


def _count_pair_frequencies(token_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for token_sequence, freq in token_counts.items():
        for pair in get_adjacent_pairs(token_sequence):
            pair_counts[pair] += freq

    return pair_counts


def _sequence_pair_counts(token_sequence: Sequence[bytes]) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for pair in get_adjacent_pairs(token_sequence):
        pair_counts[pair] += 1
    return pair_counts


def _build_pair_statistics(
    token_counts: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for token_sequence, freq in token_counts.items():
        sequence_pair_counts = _sequence_pair_counts(token_sequence)
        for pair, occurrences in sequence_pair_counts.items():
            pair_counts[pair] += occurrences * freq
            pair_to_sequences.setdefault(pair, set()).add(token_sequence)

    return pair_counts, pair_to_sequences


def _remove_sequence_from_pair_index(
    token_sequence: tuple[bytes, ...],
    pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair in _sequence_pair_counts(token_sequence):
        sequences = pair_to_sequences.get(pair)
        if sequences is None:
            continue
        sequences.discard(token_sequence)
        if not sequences:
            del pair_to_sequences[pair]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on a text corpus.

    Special tokens are added to the vocabulary and used as hard boundaries,
    but they do not participate in merge learning.
    """
    normalized_special_tokens = normalize_special_tokens(special_tokens)

    vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    existing_tokens = set(vocab.values())
    next_id = 256

    for special_token in normalized_special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in existing_tokens:
            vocab[next_id] = token_bytes
            existing_tokens.add(token_bytes)
            next_id += 1

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    token_counts = _collect_pretoken_counts(text, normalized_special_tokens)
    pair_counts, pair_to_sequences = _build_pair_statistics(token_counts)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merged_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        if merged_token not in existing_tokens:
            vocab[next_id] = merged_token
            existing_tokens.add(merged_token)
            next_id += 1

        affected_sequences = list(pair_to_sequences.get(best_pair, ()))
        updates: Counter[tuple[bytes, ...]] = Counter()

        for token_sequence in affected_sequences:
            freq = token_counts.pop(token_sequence)

            old_pair_counts = _sequence_pair_counts(token_sequence)
            for pair, occurrences in old_pair_counts.items():
                updated_count = pair_counts[pair] - occurrences * freq
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                else:
                    del pair_counts[pair]

            _remove_sequence_from_pair_index(token_sequence, pair_to_sequences)

            merged_sequence = tuple(merge_pair_in_sequence(token_sequence, best_pair))
            updates[merged_sequence] += freq

        for merged_sequence, added_freq in updates.items():
            token_counts[merged_sequence] += added_freq

        for merged_sequence, added_freq in updates.items():
            new_pair_counts = _sequence_pair_counts(merged_sequence)
            for pair, occurrences in new_pair_counts.items():
                pair_counts[pair] += occurrences * added_freq
                pair_to_sequences.setdefault(pair, set()).add(merged_sequence)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)

        self.special_tokens: list[str] = normalize_special_tokens(special_tokens)
        self.special_token_bytes: list[bytes] = [token.encode("utf-8") for token in self.special_tokens]

        existing_tokens = set(self.vocab.values())
        next_id = max(self.vocab.keys(), default=-1) + 1

        for token_bytes in self.special_token_bytes:
            if token_bytes not in existing_tokens:
                self.vocab[next_id] = token_bytes
                existing_tokens.add(token_bytes)
                next_id += 1

        self.token_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }
        self.special_token_to_id: dict[str, int] = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self._special_token_set = set(self.special_tokens)
        self._special_split_re = build_special_token_regex(self.special_tokens)
        self._special_overlap = max((len(token) for token in self.special_tokens), default=0) - 1

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: Sequence[str] | None = None,
    ) -> "Tokenizer":
        """Load a tokenizer from GPT-2 style vocab and merge files."""
        byte_decoder = {value: key for key, value in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {
            token_id: bytes([byte_decoder[ch] for ch in token])
            for token, token_id in raw_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned:
                    continue

                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue

                left, right = parts
                merges.append(
                    (
                        bytes([byte_decoder[ch] for ch in left]),
                        bytes([byte_decoder[ch] for ch in right]),
                    )
                )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken_to_bytes(self, pretoken: str) -> list[bytes]:
        return merge_pretoken(pretoken, self.merge_ranks)

    def _encode_pretoken_to_ids(self, pretoken: str) -> list[int]:
        return [self.token_to_id[part] for part in self._encode_pretoken_to_bytes(pretoken)]

    def _iter_encode_ordinary_text(self, text: str) -> Iterator[int]:
        yield from _encode_pretokens_to_ids(pretokenize_text(text), self.token_to_id, self.merge_ranks)

    def _split_stable_ordinary_tail(self, text: str) -> tuple[str, str]:
        """
        Split an ordinary-text tail into a safe prefix and a carryover suffix.

        The carryover keeps enough context for:
        1. a special token that may continue in the next chunk
        2. the final pre-token that may continue in the next chunk
        """
        if not text:
            return "", ""

        if self._special_overlap <= 0:
            matches = list(PRETOKENIZER_RE.finditer(text))
            if not matches:
                return "", text

            if matches[-1].end() == len(text):
                if len(matches) == 1:
                    return "", text
                stable_end = matches[-2].end()
                return text[:stable_end], text[stable_end:]

            stable_end = matches[-1].end()
            return text[:stable_end], text[stable_end:]

        if len(text) <= self._special_overlap:
            return "", text

        prefix_limit = len(text) - self._special_overlap
        stable_end = 0

        for match in PRETOKENIZER_RE.finditer(text):
            if match.end() <= prefix_limit:
                stable_end = match.end()
            else:
                break

        return text[:stable_end], text[stable_end:]

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []

        for is_special, unit in iter_text_units(text, self.special_tokens):
            if is_special:
                token_ids.append(self.special_token_to_id[unit])
            else:
                token_ids.extend(self._encode_pretoken_to_ids(unit))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode a stream of text chunks.

        This keeps a carryover suffix so chunk boundaries do not force token
        boundaries.
        """
        buffer = ""

        for chunk in iterable:
            if not chunk:
                continue

            buffer += chunk

            if self._special_split_re is not None:
                last_end = 0
                for match in self._special_split_re.finditer(buffer):
                    ordinary_segment = buffer[last_end : match.start()]
                    if ordinary_segment:
                        yield from self._iter_encode_ordinary_text(ordinary_segment)

                    yield self.special_token_to_id[match.group(0)]
                    last_end = match.end()

                trailing_segment = buffer[last_end:]
            else:
                trailing_segment = buffer

            stable_prefix, buffer = self._split_stable_ordinary_tail(trailing_segment)
            if stable_prefix:
                yield from self._iter_encode_ordinary_text(stable_prefix)

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: Sequence[int]) -> str:
        token_bytes = [self.vocab[token_id] for token_id in ids]
        return b"".join(token_bytes).decode("utf-8", errors="replace")
