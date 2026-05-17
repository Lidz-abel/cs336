# Benchmark Outputs

This directory contains both historical benchmark artifacts and the current formal RTX 4090 result set.

## Current Formal Result

Use this directory as the canonical current-hardware output:

- `2026-04-27_rtx4090_formal/`

Primary files:

- `2026-04-27_rtx4090_formal/RESULTS.md`: human-readable normalized report
- `2026-04-27_rtx4090_formal/summary.csv`: machine-readable task summary
- `2026-04-27_rtx4090_formal/summary.md`: raw task table
- `2026-04-27_rtx4090_formal/logs/`: per-task logs
- `2026-04-27_rtx4090_formal/memory_snapshots/`: successful memory profiler snapshots
- `2026-04-27_rtx4090_formal/manifest.json`: task manifest

Hardware for this run:

- `8 x NVIDIA GeForce RTX 4090`
- `24564 MiB` per GPU
- NVIDIA driver `590.48.01`

## Historical Results

Files prefixed with `2026-04-16_` are historical A6000 measurements. Keep them for reference only; do not mix them into the RTX 4090 writeup tables.

## Removed Dry Runs

The temporary dry-run output directories were removed after validation so the current formal run has a single unambiguous output directory.
