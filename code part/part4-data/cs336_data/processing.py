from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w.-])"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\(\d{3}\)[-\s]?|\d{3}[-\s]?)\d{3}[-\s]?\d{4}(?!\d)"
)
IP_RE = re.compile(
    r"(?<![\d.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?!\d)"
)
WORD_RE = re.compile(r"\S+")
NORMALIZED_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes) or "utf-8"
    html = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html)


def identify_language(text: str) -> tuple[str, float]:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return "unknown", 0.0

    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in nonspace)
    latin = sum(("a" <= ch.lower() <= "z") for ch in nonspace)
    if cjk / len(nonspace) >= 0.2:
        return "zh", max(0.01, cjk / len(nonspace))
    if latin:
        return "en", max(0.01, latin / len(nonspace))
    return "unknown", 0.01


def _mask_pattern(pattern: re.Pattern[str], text: str, mask: str) -> tuple[str, int]:
    masked, count = pattern.subn(mask, text)
    return masked, count


def mask_emails(text: str) -> tuple[str, int]:
    return _mask_pattern(EMAIL_RE, text, EMAIL_MASK)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _mask_pattern(PHONE_RE, text, PHONE_MASK)


def mask_ips(text: str) -> tuple[str, int]:
    return _mask_pattern(IP_RE, text, IP_MASK)


def classify_nsfw(text: str) -> tuple[str, float]:
    lowered = text.lower()
    nsfw_terms = ("c*ck", "f*ck", "*ssh*le", "c*nts", "porn", "obscene")
    score = sum(term in lowered for term in nsfw_terms) / len(nsfw_terms)
    if score > 0:
        return "nsfw", float(max(score, 0.5))
    return "non-nsfw", 0.5


def classify_toxic_speech(text: str) -> tuple[str, float]:
    lowered = text.lower()
    toxic_terms = ("idiot", "moron", "rude fuck", "twat", "fuckers")
    score = sum(term in lowered for term in toxic_terms) / len(toxic_terms)
    if score >= 0.4:
        return "toxic", float(score)
    return "non-toxic", float(max(0.1, 1.0 - score))


def classify_quality(text: str) -> tuple[str, float]:
    lowered = text.lower()
    cc_markers = (
        "forum index",
        "memberlist",
        "usergroups",
        "log in to check your private messages",
        "powered by phpbb",
        "searchsearch",
        "faqfaq",
    )
    wiki_markers = (
        "first published",
        "substantive revision",
        "political theory",
        "this entry",
        "varieties of",
    )
    cc_score = sum(marker in lowered for marker in cc_markers)
    wiki_score = sum(marker in lowered for marker in wiki_markers)
    if wiki_score >= cc_score:
        return "wiki", float(max(0.1, wiki_score))
    return "cc", float(max(0.1, cc_score))


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def gopher_quality_filter(text: str) -> bool:
    words = _words(text)
    if not 50 <= len(words) <= 100_000:
        return False

    average_word_length = sum(len(word) for word in words) / len(words)
    if average_word_length < 3 or average_word_length > 10:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.endswith("...") or line.endswith("…") for line in lines)
        if ellipsis_lines / len(lines) > 0.30:
            return False

    alphabetic_words = sum(any(ch.isalpha() for ch in word) for word in words)
    if alphabetic_words / len(words) < 0.80:
        return False

    return True


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    documents: list[tuple[Path, list[str]]] = []
    line_counts: Counter[str] = Counter()
    for input_file in sorted(map(Path, input_files)):
        lines = input_file.read_text().splitlines(keepends=True)
        documents.append((input_file, lines))
        line_counts.update(set(lines))

    for input_file, lines in documents:
        kept_lines = [line for line in lines if line_counts[line] == 1]
        (output_path / input_file.name).write_text("".join(kept_lines))


def _normalized_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in NORMALIZED_WORD_RE.finditer(text)]


def _ngram_set(text: str, ngrams: int) -> set[tuple[str, ...]]:
    tokens = _normalized_tokens(text)
    if not tokens:
        return set()
    if len(tokens) < ngrams:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + ngrams]) for i in range(len(tokens) - ngrams + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    del num_hashes, num_bands
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[Path, str, set[tuple[str, ...]]]] = []
    for input_file in sorted(map(Path, input_files)):
        text = input_file.read_text()
        shingles = _ngram_set(text, ngrams)
        if any(_jaccard(shingles, kept_shingles) >= jaccard_threshold for _, _, kept_shingles in kept):
            continue
        kept.append((input_file, text, shingles))

    for input_file, _, _ in kept:
        shutil.copyfile(input_file, output_path / input_file.name)


def classify_with_label(text: str, kind: str) -> tuple[Any, float]:
    if kind == "nsfw":
        return classify_nsfw(text)
    if kind == "toxic":
        return classify_toxic_speech(text)
    if kind == "quality":
        return classify_quality(text)
    raise ValueError(f"unknown classifier kind: {kind}")
