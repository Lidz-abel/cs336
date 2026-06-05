from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

_TOKENIZER = None
_EOS_TOKEN_ID = None


def _read_documents(input_path: Path) -> list[str]:
    paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.jsonl"))
    documents: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    documents.append(json.loads(line)["text"])
                else:
                    documents.append(line)
    return documents


def _init_tokenizer() -> None:
    global _TOKENIZER, _EOS_TOKEN_ID
    _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    _EOS_TOKEN_ID = _TOKENIZER.eos_token_id


def _tokenize_document(document: str) -> list[int]:
    if _TOKENIZER is None or _EOS_TOKEN_ID is None:
        _init_tokenizer()
    return _TOKENIZER.encode(document) + [_EOS_TOKEN_ID]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize filtered assignment4 data with the GPT-2 tokenizer.")
    parser.add_argument("--input", type=Path, required=True, help="Filtered JSONL file or directory of JSONL shards.")
    parser.add_argument("--output", type=Path, required=True, help="Output .bin path written as np.uint16.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    documents = _read_documents(args.input)

    with multiprocessing.Pool(args.workers, initializer=_init_tokenizer) as pool:
        encoded = list(
            tqdm(
                pool.imap(_tokenize_document, documents, chunksize=100),
                total=len(documents),
                desc="Tokenizing documents",
            )
        )

    all_ids = [token_id for document_ids in encoded for token_id in document_ids]
    ids_array = np.array(all_ids, dtype=np.uint16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ids_array.tofile(args.output)
    print(f"Tokenized {len(documents)} documents into {len(ids_array)} tokens at {args.output}")


if __name__ == "__main__":
    main()
