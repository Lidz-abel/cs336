from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from tldextract import TLDExtract
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

from cs336_data.processing import (
    classify_nsfw,
    classify_quality,
    classify_toxic_speech,
    gopher_quality_filter,
    identify_language,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


@dataclass
class FilterStats:
    total_records: int = 0
    kept: int = 0
    non_conversion: int = 0
    non_english: int = 0
    low_quality: int = 0
    nsfw: int = 0
    toxic: int = 0
    pii_email: int = 0
    pii_phone: int = 0
    pii_ip: int = 0


def _normalize_document(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _filter_document(text: str) -> tuple[str | None, Counter[str]]:
    counts: Counter[str] = Counter()
    text = _normalize_document(text)
    if not text:
        counts["low_quality"] += 1
        return None, counts

    language, language_score = identify_language(text)
    if language != "en" or language_score < 0.5:
        counts["non_english"] += 1
        return None, counts

    if not gopher_quality_filter(text):
        counts["low_quality"] += 1
        return None, counts

    quality_label, _ = classify_quality(text)
    if quality_label == "cc":
        counts["low_quality"] += 1
        return None, counts

    nsfw_label, _ = classify_nsfw(text)
    if nsfw_label == "nsfw":
        counts["nsfw"] += 1
        return None, counts

    toxic_label, _ = classify_toxic_speech(text)
    if toxic_label == "toxic":
        counts["toxic"] += 1
        return None, counts

    text, emails = mask_emails(text)
    text, phones = mask_phone_numbers(text)
    text, ips = mask_ips(text)
    counts["pii_email"] += emails
    counts["pii_phone"] += phones
    counts["pii_ip"] += ips
    return text, counts


def process_wet_file(input_path: Path, output_path: Path) -> FilterStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = FilterStats()
    extractor = TLDExtract(suffix_list_urls=())

    with gzip.open(input_path, "rb") as in_stream, output_path.open("w", encoding="utf-8") as out_stream:
        for record in ArchiveIterator(in_stream):
            stats.total_records += 1
            if record.rec_type != "conversion":
                stats.non_conversion += 1
                continue

            payload = record.content_stream().read()
            text = payload.decode("utf-8", errors="replace")
            filtered, counts = _filter_document(text)
            for key, value in counts.items():
                setattr(stats, key, getattr(stats, key) + value)
            if filtered is None:
                continue

            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            domain = extractor(url).registered_domain
            out_stream.write(json.dumps({"url": url, "domain": domain, "text": filtered}, ensure_ascii=False) + "\n")
            stats.kept += 1

    return stats


def _merge_stats(stats: list[FilterStats]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for item in stats:
        merged.update(asdict(item))
    return dict(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter English Common Crawl WET files into JSONL training text.")
    parser.add_argument("--input-dir", type=Path, default=Path("/shared-data/english-wet-data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.warc.wet.gz"))
    if args.limit is not None:
        input_files = input_files[: args.limit]
    if not input_files:
        raise FileNotFoundError(f"no .warc.wet.gz files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for input_path in input_files:
            output_path = args.output_dir / f"{input_path.name}.jsonl"
            futures.append(executor.submit(process_wet_file, input_path, output_path))

        stats = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtering WET files"):
            stats.append(future.result())

    summary = _merge_stats(stats)
    (args.output_dir / "filter_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
