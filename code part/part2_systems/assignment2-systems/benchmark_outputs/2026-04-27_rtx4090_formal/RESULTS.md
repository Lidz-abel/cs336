# RTX 4090 Formal Benchmark Results

This document is the canonical human-readable record for the current-hardware benchmark run.

## Canonical Outputs

- Run directory: `benchmark_outputs/2026-04-27_rtx4090_formal/`
- Machine-readable summary: `summary.csv`
- Raw task table: `summary.md`
- Per-task logs: `logs/*.log`
- Successful memory snapshots: `memory_snapshots/*.pickle`
- Task manifest: `manifest.json`

Dry-run directories were removed after validation so this run directory is the unique current-hardware output entry.

## Hardware And Run Scope

- Hardware: `8 x NVIDIA GeForce RTX 4090`
- GPU memory: `24564 MiB` per GPU
- NVIDIA driver: `590.48.01`
- CPU: `Intel(R) Xeon(R) Gold 6430`, `128` logical CPUs
- System memory: `1.0 TiB` total, approximately `987 GiB` available at the pre-run check
- Benchmark model batch size: `4`
- Benchmark vocabulary size: `10000`
- Main context length: `128`
- Baseline and BF16 timing: `5` warmup steps, `10` measured steps
- Memory profiling timing: `1` warmup step, `1` measured step

All tasks were scheduled through the 8-GPU worker queue in `run_current_hardware_benchmarks.py`. OOM outcomes are recorded as hardware-bound results, not script failures.

## Phase Completion Summary

| phase | PASS | OOM | FAIL |
|---|---:|---:|---:|
| `01_baseline_fp32` | 9 | 1 | 0 |
| `02_warmup_sweep_fp32` | 36 | 4 | 0 |
| `03_mixed_precision_bf16` | 9 | 1 | 0 |
| `04_memory_profiling` | 6 | 6 | 0 |

## 1. FP32 Baseline Timing

| model | forward mean ms | forward std ms | forward+backward mean ms | forward+backward std ms |
|---|---:|---:|---:|---:|
| small | 11.448 | 0.066 | 35.383 | 0.514 |
| medium | 23.370 | 0.088 | 68.459 | 1.153 |
| large | 39.026 | 0.123 | 120.142 | 8.285 |
| xl | 73.647 | 0.129 | 200.166 | 1.484 |
| 2.7b | 107.876 | 1.129 | OOM | - |

The `2.7b` forward pass fits on a 24 GB RTX 4090 at context length 128, but `2.7b` forward+backward does not fit. This OOM boundary differs from the earlier A6000 runs because RTX 4090 has roughly half the GPU memory per card.

## 2. Warmup Sweep

| model | mode | warmup 0 mean ms | warmup 1 mean ms | warmup 2 mean ms | warmup 5 mean ms |
|---|---|---:|---:|---:|---:|
| small | forward | 37.886 | 11.262 | 12.244 | 11.693 |
| small | forward_backward | 84.955 | 45.442 | 39.588 | 37.287 |
| medium | forward | 50.364 | 23.359 | 23.370 | 26.466 |
| medium | forward_backward | 108.381 | 72.443 | 74.270 | 75.677 |
| large | forward | 66.149 | 38.941 | 38.116 | 38.014 |
| large | forward_backward | 218.791 | 113.397 | 115.380 | 120.992 |
| xl | forward | 104.379 | 73.589 | 74.780 | 73.473 |
| xl | forward_backward | 319.744 | 205.386 | 201.498 | 202.771 |
| 2.7b | forward | 135.219 | 111.507 | 107.756 | 109.525 |
| 2.7b | forward_backward | OOM | OOM | OOM | OOM |

Warmup matters materially. Runs with `0` warmup frequently include first-use CUDA/kernel overhead and show high variance; `1-2` warmup steps usually remove most of that effect, while `5` warmup steps is the stable handout baseline.

## 3. BF16 Mixed Precision Comparison

| model | mode | FP32 mean ms | BF16 mean ms | speedup FP32/BF16 |
|---|---|---:|---:|---:|
| small | forward | 11.448 | 13.897 | 0.82x |
| small | forward_backward | 35.383 | 45.010 | 0.79x |
| medium | forward | 23.370 | 26.436 | 0.88x |
| medium | forward_backward | 68.459 | 85.706 | 0.80x |
| large | forward | 39.026 | 41.213 | 0.95x |
| large | forward_backward | 120.142 | 135.586 | 0.89x |
| xl | forward | 73.647 | 55.427 | 1.33x |
| xl | forward_backward | 200.166 | 192.616 | 1.04x |
| 2.7b | forward | 107.876 | 55.777 | 1.93x |
| 2.7b | forward_backward | OOM | OOM | - |

BF16 improves the larger forward-only runs substantially (`xl` and `2.7b`), but it is not uniformly faster for smaller models or forward+backward in this implementation. The `2.7b` forward+backward case remains OOM even with BF16 autocast because parameters and optimizer/state-relevant tensors are still not fully reduced to a memory-saving training configuration.

## 4. Memory Profiling

| context | precision | forward mean ms | peak allocated MB | peak reserved MB | snapshot |
|---:|---|---:|---:|---:|---|
| 128 | none | 111.196 | 13237.769 | 13266.000 | `04_memory_profiling_size_2p7b_mode_forward_ctx_128_warmup_1_mp_none.pickle` |
| 128 | bf16 | 57.257 | 13237.769 | 13270.000 | `04_memory_profiling_size_2p7b_mode_forward_ctx_128_warmup_1_mp_bf16.pickle` |
| 256 | none | 211.837 | 13342.996 | 13536.000 | `04_memory_profiling_size_2p7b_mode_forward_ctx_256_warmup_1_mp_none.pickle` |
| 256 | bf16 | 96.380 | 13322.996 | 13420.000 | `04_memory_profiling_size_2p7b_mode_forward_ctx_256_warmup_1_mp_bf16.pickle` |
| 512 | none | 482.670 | 13873.357 | 13938.000 | `04_memory_profiling_size_2p7b_mode_forward_ctx_512_warmup_1_mp_none.pickle` |
| 512 | bf16 | 234.209 | 13799.357 | 13928.000 | `04_memory_profiling_size_2p7b_mode_forward_ctx_512_warmup_1_mp_bf16.pickle` |

| context | precision | train_step status | log |
|---:|---|---|---|
| 128 | none | OOM | `04_memory_profiling_size_2p7b_mode_train_step_ctx_128_warmup_1_mp_none.log` |
| 128 | bf16 | OOM | `04_memory_profiling_size_2p7b_mode_train_step_ctx_128_warmup_1_mp_bf16.log` |
| 256 | none | OOM | `04_memory_profiling_size_2p7b_mode_train_step_ctx_256_warmup_1_mp_none.log` |
| 256 | bf16 | OOM | `04_memory_profiling_size_2p7b_mode_train_step_ctx_256_warmup_1_mp_bf16.log` |
| 512 | none | OOM | `04_memory_profiling_size_2p7b_mode_train_step_ctx_512_warmup_1_mp_none.log` |
| 512 | bf16 | OOM | `04_memory_profiling_size_2p7b_mode_train_step_ctx_512_warmup_1_mp_bf16.log` |

For the `2.7b` model on a 24 GB RTX 4090, forward memory profiling succeeds at context lengths 128, 256, and 512. Full `train_step` memory profiling OOMs at all three context lengths for both FP32 and BF16.

## OOM Boundary

- `2.7b forward_backward`, context length `128`, batch size `4`, vocab size `10000`: OOM in FP32 and BF16.
- `2.7b train_step`, context lengths `128`, `256`, `512`: OOM in FP32 and BF16.
- OOM logs are preserved under `logs/` and should be cited when explaining why some handout measurements cannot be produced on 24 GB cards.

## Recommended Citation In Writeup

Use this run as the formal current-hardware result set. If old A6000 measurements are referenced, label them as historical only and do not mix them into the RTX 4090 tables.
