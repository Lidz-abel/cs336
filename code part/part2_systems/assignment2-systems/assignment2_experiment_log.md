# Assignment 2 Experiment Log

## 2026-04-10 Phase 0: Fresh Clone Sanity Check

- Status: PASS
- Purpose: Confirm that the newly cloned `assignment2-systems` repo is clean before any environment setup or implementation work.

### Commands

```bash
pwd
git status --short
ls -la "/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems"
find "/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems" -maxdepth 2 -type d \( -name "llmPart2" -o -name ".venv" \)
```

### Key Results

- Workspace root: `/home/u-lidz/cs336/cs336`
- Repo status: clean (`git status --short` returned no tracked changes)
- Fresh clone confirmed for `assignment2-systems`
- No existing `llmPart2` or project-local `.venv` was present

### Next Step

- Create a fresh virtual environment named `llmPart2`
- Configure it from `pyproject.toml` and `uv.lock`

## 2026-04-10 Phase 1: Hardware Inventory

- Status: PASS
- Purpose: Record the hardware context for all later correctness and performance experiments.

### Commands

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu
free -h
df -h .
```

### Key Results

- GPU: 4 x `NVIDIA RTX A6000`
- GPU memory: `49140 MiB` per GPU
- NVIDIA driver: `535.274.02`
- CPU: `Intel(R) Xeon(R) w7-3455`
- CPU topology: 24 physical cores / 48 logical CPUs
- System memory: `1.0 TiB`
- Available disk at environment setup time: `1.2T`

### Experiment Notes

- All future GPU performance numbers must state the exact GPU count, dtype, batch size, and whether the run is single-GPU or multi-GPU.
- CPU and GPU timings must not be mixed into the same performance conclusion.

## 2026-04-10 Phase 2: Create `llmPart2`

- Status: PASS
- Purpose: Create a fresh isolated environment for assignment2.

### Command

```bash
/usr/bin/time -p python3 -m venv llmPart2
```

### Result

- Environment created successfully at `assignment2-systems/llmPart2`
- Timing:
  - `real 2.91`
  - `user 2.56`
  - `sys 0.19`

## 2026-04-10 Phase 3: Install `uv` Into `llmPart2`

- Status: PASS
- Purpose: Install `uv` into the new virtual environment so dependency resolution follows the project's own packaging flow.

### Command

```bash
/usr/bin/time -p llmPart2/bin/python -m pip install uv
```

### Result

- Installed `uv 0.11.6`
- Timing:
  - `real 2.09`
  - `user 1.39`
  - `sys 0.18`

## 2026-04-10 Phase 4: Sync Environment From Project Metadata

- Status: PASS
- Purpose: Configure `llmPart2` from `pyproject.toml` and `uv.lock`, without manually selecting package versions.

### Command

```bash
source llmPart2/bin/activate && /usr/bin/time -p uv sync --frozen --active
```

### Result

- Sync completed successfully
- Installed key packages:
  - `torch==2.6.0`
  - `triton==3.2.0`
  - `einops==0.8.1`
  - `pytest==8.4.1`
  - editable `cs336-basics`
  - editable `cs336-systems`
- Timing:
  - `real 1.32`
  - `user 0.57`
  - `sys 0.44`

### Notes

- The sync completed unusually quickly.
- Inference: the machine likely reused a preexisting local `uv` package cache, because the command reported installs without large download output.

## 2026-04-10 Phase 5: Validate `llmPart2`

- Status: PASS
- Purpose: Confirm that the new environment is the correct baseline environment for assignment2.

### Commands

```bash
llmPart2/bin/python -V
llmPart2/bin/uv --version
llmPart2/bin/python -m pytest --version
'/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/llmPart2/bin/python' -c "import cs336_basics, cs336_systems, torch, triton, einops; print({'cs336_basics': getattr(cs336_basics, '__file__', None), 'cs336_systems': getattr(cs336_systems, '__file__', None), 'torch': torch.__version__, 'triton': triton.__version__, 'einops': getattr(einops, '__version__', 'no __version__'), 'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})"
du -sh llmPart2
```

### Key Results

- Python: `3.11.7`
- `uv`: `0.11.6`
- `pytest`: `8.4.1`
- `torch`: `2.6.0+cu124`
- `triton`: `3.2.0`
- `einops`: `0.8.1`
- `cs336_basics`: importable
- `cs336_systems`: importable
- CUDA availability: `True`
- CUDA device count: `4`
- Visible CUDA devices:
  - `NVIDIA RTX A6000`
  - `NVIDIA RTX A6000`
  - `NVIDIA RTX A6000`
  - `NVIDIA RTX A6000`
- Environment size: `5.5G`

### Current Baseline State

- Code remains unimplemented for assignment2
- Environment is now ready for clean baseline testing

### Next Step

- Run the official baseline tests inside `llmPart2`
- Record failures separately as:
  - environment/setup failures
  - expected `NotImplementedError` / unimplemented-interface failures

## 2026-04-10 Phase 6: Official Baseline Test Run In `llmPart2`

- Status: FAIL
- Purpose: Capture the clean initial failure surface of assignment2 before writing any implementation code.

### Command

```bash
source llmPart2/bin/activate && /usr/bin/time -p python -m pytest tests -q
```

### Timing

- `real 24.93`
- `user 103.53`
- `sys 6.20`

### Summary

- Total result: `16 failed, 1 warning in 23.08s`
- No environment or dependency failure occurred during collection or execution
- All failures were due to expected unimplemented assignment interfaces

### Failure Breakdown

- FlashAttention tests failed: `6`
  - `get_flashattention_autograd_function_pytorch`
  - `get_flashattention_autograd_function_triton`
- Bucketed DDP tests failed: `6`
  - `get_ddp_bucketed`
- Individual-parameter DDP tests failed: `2`
  - `get_ddp_individual_parameters`
- Sharded optimizer tests failed: `2`
  - `get_sharded_optimizer`

### Root Cause

- `tests/adapters.py` still contains the starter `raise NotImplementedError` bodies for all assignment2 interfaces.
- This is the expected baseline state for a fresh clone with no student implementation.

### Important Observation

- The distributed tests now fail at the intended student entry points rather than at environment setup.
- This means the rebuilt `llmPart2` environment is valid for correctness work.

### Warning Observed

- One CUDA warning was emitted during Triton-related backward testing:
  - PyTorch reported that there was no current CUDA context and it attempted to set the primary context before cuBLAS use.
- This warning did not block execution and is not currently treated as a baseline failure.

### Next Step

- Begin implementation with `get_flashattention_autograd_function_pytorch`
- Keep all later test results comparable against this baseline

## 2026-04-16 Phase 7: Rebuild `llmPart2` Against Custom `cs336-basics`

- Status: PASS
- Purpose: Rebuild the assignment2 environment so it stably uses the custom assignment1 implementation under `cs336-basics/`, rather than relying on accidental import-path behavior.

### Precondition Fix

- Updated assignment2 root `pyproject.toml` so `[tool.uv.sources]` points to:
  - `cs336-basics = { path = "./cs336-basics", editable = true }`

### Initial Failure

- Attempted command:

```bash
/usr/bin/time -p python3 -m venv llmPart2
```

- Result: FAIL
- Observed error:
  - `ensurepip is not available`

### Root Cause

- The system `python3` at `/usr/bin/python3` could not bootstrap a working venv with `ensurepip` on this machine.

### Fix

- Installed `virtualenv` using the base Anaconda Python, with trusted hosts due a PyPI TLS failure:

```bash
/home/u-lidz/anaconda3/bin/python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org virtualenv
```

- Removed the broken partial environment:

```bash
rm -rf "/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/llmPart2"
```

- Recreated the environment with `virtualenv`:

```bash
/usr/bin/time -p /home/u-lidz/anaconda3/bin/python3 -m virtualenv llmPart2
```

- Installed `uv` into the rebuilt environment:

```bash
'/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/llmPart2/bin/python' -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org uv
```

- Regenerated the lockfile and resynced:

```bash
source llmPart2/bin/activate && /usr/bin/time -p uv lock
source llmPart2/bin/activate && /usr/bin/time -p uv sync --active
```

### Key Results

- `virtualenv` environment creation succeeded
  - `real 0.52`
  - `user 0.42`
  - `sys 0.14`
- `uv lock` succeeded against the updated package layout
  - `real 0.35`
  - `user 0.24`
  - `sys 0.12`
- `uv sync --active` succeeded
  - `real 1.36`
  - `user 0.81`
  - `sys 0.67`

### Validation

- `cs336_basics` now imports from the custom package path:
  - `/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/cs336-basics/cs336_basics/__init__.py`
- Custom model/optimizer import check passed:

```bash
from cs336_basics.transformer_lm import *
from cs336_basics.AdamW import AdamW
```

- Environment versions after rebuild:
  - `torch 2.6.0+cu124`
  - `triton 3.2.0`
  - `cuda_available = True`
  - `cuda_device_count = 4`

### Outcome

- The environment now stably resolves to the custom assignment1 implementation and is ready for subsequent benchmarking and systems tasks.

## 2026-04-16 Phase 8: Add Thin `nn.Module` Wrappers Around Custom Basics Functions

- Status: PASS
- Purpose: Add a wrapper layer that makes the custom assignment1 transformer usable as a standard PyTorch module for benchmarking and later systems experiments, without replacing any of the handwritten model math.

### Code Changes

- Added [cs336-basics/cs336_basics/model.py](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/cs336-basics/cs336_basics/model.py)
- Updated [cs336-basics/cs336_basics/__init__.py](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/cs336-basics/cs336_basics/__init__.py)

### Implementation Boundary

- The new wrappers use `nn.Module` and `nn.Parameter` only as containers.
- Forward computation still dispatches into the custom handwritten functions:
  - `embedding.py`
  - `linear.py`
  - `rmsnorm.py`
  - `swiglu.py`
  - `multihead_self_attention_with_rope.py`
  - `transformer_block.py`
  - `transformer_lm.py`
- No `nn.Linear`, `nn.MultiheadAttention`, or `nn.Transformer` was introduced into the model forward path.

### Validation Commands

```bash
rg -n "from cs336_basics|torch\.nn\.Linear|MultiheadAttention|Transformer\(" 'cs336-basics/cs336_basics/model.py'
source llmPart2/bin/activate && python -c "import torch; from cs336_basics import BasicsTransformerLM; model = BasicsTransformerLM(vocab_size=128, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64); x = torch.randint(0, 128, (2, 16)); y = model(x); print({'shape': tuple(y.shape), 'dtype': str(y.dtype), 'device': str(y.device), 'params': model.get_num_params(non_embedding=False)})"
source llmPart2/bin/activate && python -c "import torch; from cs336_basics import BasicsTransformerLM; from cs336_basics.AdamW import AdamW; model = BasicsTransformerLM(vocab_size=128, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64); opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01); x = torch.randint(0, 128, (2, 16)); targets = torch.randint(0, 128, (2, 16)); logits = model(x); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), targets.reshape(-1)); loss.backward(); opt.step(); opt.zero_grad(); print({'loss': float(loss), 'grad_ok': all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'param_sum': float(sum(p.detach().abs().sum() for p in model.parameters()))})"
```

### Key Results

- Import scan showed that `model.py` depends only on local `cs336_basics` functional modules for forward computation.
- Minimal forward pass succeeded in `llmPart2`.
- Minimal training step with custom `AdamW` also succeeded in `llmPart2`.
- Output summary:
  - shape: `(2, 16, 128)`
  - dtype: `torch.float32`
  - device: `cpu`
  - parameter count: `28832`
  - loss: `5.206953048706055`
  - finite gradients: `True`
  - post-step parameter absolute sum: `6293.93701171875`

### Outcome

- The custom basics package now exposes a usable `BasicsTransformerLM` module wrapper while keeping the handwritten function implementation as the only computation path.
- This is the correct base for the `benchmarking_script` task.

## 2026-04-16 Phase 9: Implement `benchmark.py` For Custom Basics Model

- Status: PASS
- Purpose: Create the first reusable benchmarking script for the custom `BasicsTransformerLM`, using the custom wrapper class and custom `AdamW`.

### Code Changes

- Added [benchmark.py](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark.py)

### Script Scope

- Uses the custom `BasicsTransformerLM` wrapper from `cs336_basics`
- Uses the custom `AdamW` implementation for `train_step`
- Uses the custom `cross_entropy` implementation for loss computation
- Supports three benchmark modes:
  - `forward`
  - `forward_backward`
  - `train_step`
- Supports:
  - model size presets
  - configurable batch size / context length / vocab size
  - warmup steps
  - measured steps
  - CUDA synchronization when running on GPU
  - per-step timing output
  - mean / std / min / max summary

### Validation Commands

```bash
source llmPart2/bin/activate && python benchmark.py --model-size small --mode train_step --device cpu --batch-size 2 --context-length 32 --warmup-steps 1 --measurement-steps 2
source llmPart2/bin/activate && python benchmark.py --model-size small --mode forward --device cuda --batch-size 2 --context-length 32 --warmup-steps 1 --measurement-steps 2
```

### Key Results

- CPU `train_step` validation passed
  - parameter count: `162417408`
  - warmup: `423.743 ms`
  - measured steps:
    - `248.244 ms`
    - `245.435 ms`
  - mean: `246.839 ms`
  - std: `1.987 ms`
  - throughput: `259.278 tokens/s`

- CUDA `forward` validation passed
  - device: `NVIDIA RTX A6000`
  - parameter count: `162417408`
  - warmup: `1446.140 ms`
  - measured steps:
    - `10.868 ms`
    - `11.117 ms`
  - mean: `10.993 ms`
  - std: `0.176 ms`
  - throughput: `5822.116 tokens/s`

### Notes

- The default benchmark `vocab_size` is currently `32000`, so even the `small` preset has a large parameter count because both token embeddings and LM head are untied.
- The script intentionally reuses one fixed random batch so the timing reflects model execution rather than repeated input generation overhead.

### Outcome

- The project now has a working baseline benchmarking script built entirely on the custom assignment1 model path.
- This script is ready to be extended later for mixed precision, `torch.compile`, profiling, and memory experiments.

## 2026-04-16 Phase 10: Formal Handout Baseline Timing Sweep

- Status: PASS
- Purpose: Run the first formal baseline timing sweep requested by `benchmarking_script` using the handout model-size presets.

### Assumption

- The handout fixes `vocab_size=10000`, `batch_size=4`, `warmup_steps=5`, and `measurement_steps=10`, but does not pin a single `context_length` in §1.1.3(b).
- For this first formal baseline group, we used:
  - `context_length=128`
  - single GPU
  - `device=cuda`
  - one `NVIDIA RTX A6000`

### Formal Sweep Command

```bash
mkdir -p benchmark_outputs && source llmPart2/bin/activate && log_file='benchmark_outputs/2026-04-16_baseline_ctx128_cuda.log' && : > "$log_file" && for size in small medium large xl 2.7b; do for mode in forward forward_backward; do echo "=== size=$size mode=$mode ===" | tee -a "$log_file"; /usr/bin/time -p python benchmark.py --model-size "$size" --mode "$mode" --device cuda --batch-size 4 --vocab-size 10000 --context-length 128 --warmup-steps 5 --measurement-steps 10 2>&1 | tee -a "$log_file"; done; done
```

### Raw Output And Parsed Tables

- Raw sweep log:
  - [benchmark_outputs/2026-04-16_baseline_ctx128_cuda.log](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx128_cuda.log)
- Parsed CSV summary:
  - [benchmark_outputs/2026-04-16_baseline_ctx128_cuda_summary.csv](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx128_cuda_summary.csv)
- Parsed Markdown summary:
  - [benchmark_outputs/2026-04-16_baseline_ctx128_cuda_summary.md](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx128_cuda_summary.md)

### Parsing Command

```bash
source llmPart2/bin/activate && python - <<'PY'
from pathlib import Path
import csv
import re

log_path = Path('benchmark_outputs/2026-04-16_baseline_ctx128_cuda.log')
text = log_path.read_text()
pattern = re.compile(
    r"=== size=(?P<size>[^ ]+) mode=(?P<mode>[^=\n]+) ===\n"
    r".*?summary\n"
    r"  mean_ms=(?P<mean>[0-9.]+)\n"
    r"  std_ms=(?P<std>[0-9.]+)\n"
    r"  min_ms=(?P<min>[0-9.]+)\n"
    r"  max_ms=(?P<max>[0-9.]+)",
    re.S,
)
...
PY
```

### Key Results

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | approx backward mean (ms) |
|---|---:|---:|---:|---:|---:|
| small | 13.122 | 0.037 | 38.909 | 0.428 | 25.787 |
| medium | 39.417 | 0.072 | 115.280 | 0.556 | 75.863 |
| large | 75.705 | 0.509 | 221.761 | 0.826 | 146.056 |
| xl | 153.255 | 0.954 | 438.801 | 0.908 | 285.546 |
| 2.7b | 222.738 | 0.602 | 634.155 | 1.140 | 411.417 |

### Observations

- All five handout model sizes fit on a single RTX A6000 at `batch_size=4`, `context_length=128`, `vocab_size=10000`.
- No OOM or runtime failure occurred during the sweep.
- Standard deviations were small across all runs, so after warmup the timing was stable.
- The backward portion is consistently more expensive than the forward portion, and both scale up smoothly with model size.

### Minor Tooling Issue

- An initial parsing helper failed with `/bin/bash: line 1: python: command not found`.
- Root cause: the parsing shell was not using the `llmPart2` interpreter.
- Fix: reran the parsing step under `source llmPart2/bin/activate`.

### Outcome

- We now have the first formally logged handout baseline timing group, with raw logs and reusable summary tables.
- This provides the timing baseline for the later warmup comparison, mixed precision, `torch.compile`, and profiler sections.

## 2026-04-16 Phase 11: Extend Formal Baseline Timing To Larger Context Lengths

- Status: PASS
- Purpose: Extend the formal CUDA baseline sweep from `context_length=128` to larger context lengths while keeping the rest of the handout benchmarking setup fixed.

### Fixed Benchmarking Setup

- device: single `NVIDIA RTX A6000`
- `batch_size=4`
- `vocab_size=10000`
- `warmup_steps=5`
- `measurement_steps=10`
- modes:
  - `forward`
  - `forward_backward`
- context lengths added:
  - `256`
  - `512`
  - `1024`

### Sweep Command

```bash
source llmPart2/bin/activate && mkdir -p benchmark_outputs && set +e
for ctx in 256 512 1024; do
  log_file="benchmark_outputs/2026-04-16_baseline_ctx${ctx}_cuda.log"
  : > "$log_file"
  for size in small medium large xl 2.7b; do
    for mode in forward forward_backward; do
      echo "=== context=$ctx size=$size mode=$mode ===" | tee -a "$log_file"
      /usr/bin/time -p python benchmark.py --model-size "$size" --mode "$mode" --device cuda --batch-size 4 --vocab-size 10000 --context-length "$ctx" --warmup-steps 5 --measurement-steps 10 2>&1 | tee -a "$log_file"
      status=${PIPESTATUS[0]}
      echo "exit_status=$status" | tee -a "$log_file"
    done
  done
done
```

### Raw Output And Parsed Tables

- Raw logs:
  - [benchmark_outputs/2026-04-16_baseline_ctx256_cuda.log](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx256_cuda.log)
  - [benchmark_outputs/2026-04-16_baseline_ctx512_cuda.log](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx512_cuda.log)
  - [benchmark_outputs/2026-04-16_baseline_ctx1024_cuda.log](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx1024_cuda.log)
- Combined CSV summary:
  - [benchmark_outputs/2026-04-16_baseline_ctx256_512_1024_cuda_summary.csv](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx256_512_1024_cuda_summary.csv)
- Combined Markdown summary:
  - [benchmark_outputs/2026-04-16_baseline_ctx256_512_1024_cuda_summary.md](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_baseline_ctx256_512_1024_cuda_summary.md)

### Key Results

#### `context_length=256`

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | status |
|---|---:|---:|---:|---:|---|
| small | 26.219 | 0.133 | 81.132 | 0.130 | PASS |
| medium | 78.323 | 0.052 | 235.239 | 0.908 | PASS |
| large | 160.916 | 1.282 | 474.260 | 0.725 | PASS |
| xl | 319.407 | 1.207 | 930.446 | 1.806 | PASS |
| 2.7b | 474.254 | 1.349 | 1288.198 | 1.971 | PASS |

#### `context_length=512`

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | status |
|---|---:|---:|---:|---:|---|
| small | 62.471 | 0.167 | 195.324 | 0.212 | PASS |
| medium | 185.372 | 0.154 | 564.280 | 0.519 | PASS |
| large | 371.629 | 0.778 | 1112.590 | 1.037 | PASS |
| xl | 719.213 | 0.566 | 2104.343 | 1.612 | PASS |
| 2.7b | 1025.964 | 0.856 | - | - | `forward=PASS`, `forward_backward=OOM` |

#### `context_length=1024`

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | status |
|---|---:|---:|---:|---:|---|
| small | 168.471 | 0.119 | 535.220 | 0.165 | PASS |
| medium | 484.431 | 0.406 | 1507.564 | 0.893 | PASS |
| large | 986.346 | 0.655 | - | - | `forward=PASS`, `forward_backward=OOM` |
| xl | 1797.081 | 1.761 | - | - | `forward=PASS`, `forward_backward=OOM` |
| 2.7b | 2336.771 | 2.753 | - | - | `forward=PASS`, `forward_backward=OOM` |

### OOM Boundary Observations

- At `context_length=256`, all five model sizes completed both `forward` and `forward_backward`.
- At `context_length=512`, the first OOM boundary appears at:
  - `2.7b` with `forward_backward`
- At `context_length=1024`, the following still fit for `forward_backward`:
  - `small`
  - `medium`
- At `context_length=1024`, the following fit only for `forward`, but OOM on `forward_backward`:
  - `large`
  - `xl`
  - `2.7b`

### Representative OOM Messages

- `context_length=512`, `2.7b`, `forward_backward`
  - OOM while allocating during RoPE application
  - requested allocation: `10.00 MiB`
- `context_length=1024`, `large`, `forward_backward`
  - OOM during softmax attention path
  - requested allocation: `320.00 MiB`
- `context_length=1024`, `xl`, `forward_backward`
  - OOM during attention masking path
  - requested allocation: `400.00 MiB`
- `context_length=1024`, `2.7b`, `forward_backward`
  - OOM during softmax normalization
  - requested allocation: `512.00 MiB`

### Observations

- Timing variance remained low even at larger sequence lengths; the post-warmup runs were still stable.
- Runtime grows substantially with context length, especially once attention cost becomes more dominant.
- The first hard memory limit appears in the largest model at `512`, and by `1024` only the smaller models still support full backward passes under this setup.

### Outcome

- The formal baseline now covers `context_length=128`, `256`, `512`, and `1024`.
- We also now have a concrete single-GPU memory boundary map for the custom baseline implementation, which will be useful for later mixed-precision and memory-profiling experiments.

## 2026-04-16 Phase 12: Add Warmup Sweep Support And Run Formal Warmup Comparison

- Status: PASS
- Purpose: Extend `benchmark.py` so the script can compare `warmup_steps=0/1/2/5` under the same benchmarking configuration, then run the formal warmup comparison for the `context_length=128` baseline group.

### Code Changes

- Updated [benchmark.py](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark.py)

### Implementation Notes

- Added `--warmup-sweep` to `benchmark.py`.
- The sweep is implemented by launching one fresh subprocess per warmup value, instead of reusing a single Python process.
- This avoids contaminating the `warmup=0` case with already-initialized CUDA runtime state from an earlier run in the same process.

### Minimal Validation

```bash
source llmPart2/bin/activate && python benchmark.py --model-size small --mode forward --device cuda --batch-size 2 --vocab-size 10000 --context-length 64 --measurement-steps 2 --warmup-sweep 0 1
```

### Minimal Validation Result

- `warmup=0`:
  - mean: `198.234 ms`
  - std: `269.004 ms`
- `warmup=1`:
  - mean: `7.357 ms`
  - std: `0.132 ms`

### Formal Warmup Sweep Setup

- device: single `NVIDIA RTX A6000`
- `batch_size=4`
- `vocab_size=10000`
- `context_length=128`
- `measurement_steps=10`
- warmup values:
  - `0`
  - `1`
  - `2`
  - `5`
- model sizes:
  - `small`
  - `medium`
  - `large`
  - `xl`
  - `2.7b`
- modes:
  - `forward`
  - `forward_backward`

### Formal Sweep Command

```bash
source llmPart2/bin/activate && mkdir -p benchmark_outputs && log_file='benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda.log' && : > "$log_file" && for size in small medium large xl 2.7b; do for mode in forward forward_backward; do echo "=== size=$size mode=$mode ===" | tee -a "$log_file"; /usr/bin/time -p python benchmark.py --model-size "$size" --mode "$mode" --device cuda --batch-size 4 --vocab-size 10000 --context-length 128 --measurement-steps 10 --warmup-sweep 0 1 2 5 2>&1 | tee -a "$log_file"; done; done
```

### Raw Output And Parsed Tables

- Raw warmup sweep log:
  - [benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda.log](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda.log)
- Parsed CSV summary:
  - [benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda_summary.csv](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda_summary.csv)
- Parsed Markdown summary:
  - [benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda_summary.md](/home/u-lidz/cs336/cs336/code part/part2_systems/assignment2-systems/benchmark_outputs/2026-04-16_warmup_sweep_ctx128_cuda_summary.md)

### Key Results

#### `mode=forward`

| size | w=0 mean/std (ms) | w=1 mean/std (ms) | w=2 mean/std (ms) | w=5 mean/std (ms) |
|---|---|---|---|---|
| small | 60.903 / 151.263 | 12.842 / 0.090 | 12.815 / 0.106 | 12.910 / 0.070 |
| medium | 74.847 / 111.606 | 39.546 / 0.054 | 39.624 / 0.035 | 39.481 / 0.025 |
| large | 110.507 / 109.593 | 76.099 / 0.185 | 76.357 / 0.643 | 76.615 / 0.815 |
| xl | 188.836 / 105.336 | 155.877 / 1.328 | 156.109 / 1.168 | 156.591 / 0.943 |
| 2.7b | 264.665 / 114.030 | 228.398 / 1.394 | 228.639 / 1.340 | 228.654 / 0.950 |

#### `mode=forward_backward`

| size | w=0 mean/std (ms) | w=1 mean/std (ms) | w=2 mean/std (ms) | w=5 mean/std (ms) |
|---|---|---|---|---|
| small | 93.550 / 173.163 | 39.398 / 1.213 | 39.477 / 0.326 | 39.579 / 1.062 |
| medium | 157.375 / 135.976 | 114.574 / 0.428 | 115.355 / 0.343 | 115.173 / 0.190 |
| large | 265.418 / 131.005 | 225.028 / 2.046 | 225.086 / 1.135 | 225.764 / 0.934 |
| xl | 494.510 / 152.238 | 447.727 / 1.263 | 449.149 / 0.863 | 449.949 / 0.677 |
| 2.7b | 685.258 / 131.250 | 645.379 / 2.664 | 647.601 / 0.874 | 650.568 / 0.958 |

### Observations

- `warmup=0` consistently produces much larger means and much larger standard deviations, because the first measured step includes cold-start overhead.
- Moving from `warmup=0` to `warmup=1` is the main correction; the timings collapse toward the stable post-warmup regime immediately.
- `warmup=1`, `warmup=2`, and `warmup=5` are close, but not identical.
- The difference between `warmup=1/2/5` is small relative to the jump from `warmup=0`, but it is still measurable, especially on larger models and backward passes.

### Outcome

- The benchmark script now natively supports rigorous warmup comparisons.
- We now have the data needed to answer handout `1.1.3(c)` with actual measurements instead of anecdotal explanation.

## 2026-04-21 Phase 13: Current Hardware Re-Inventory

- Status: PASS
- Purpose: Re-record the current machine hardware for all future assignment2 experiments, without rerunning previously completed benchmark work.

### Commands

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
lscpu
free -h
df -h "/home/lidz/cs336/code part/part2_systems/assignment2-systems"
```

### Key Results

- GPU: `8 x NVIDIA GeForce RTX 4090`
- GPU memory: `24564 MiB` per GPU
- NVIDIA driver: `590.48.01`
- CPU: `Intel(R) Xeon(R) Gold 6430`
- CPU topology: `64` physical cores / `128` logical CPUs / `2` sockets
- System memory: `1.0 TiB`
- Available memory at inventory time: `993 GiB`
- Assignment workspace filesystem: `/dev/sdb1`
- Available disk for `/home`: `14T`

### Comparison With 2026-04-10 Inventory

- This machine is not the same as the one recorded on `2026-04-10`.
- Previous inventory:
  - `4 x NVIDIA RTX A6000`, `49140 MiB` each
  - `Intel(R) Xeon(R) w7-3455`
- Current inventory:
  - `8 x NVIDIA GeForce RTX 4090`, `24564 MiB` each
  - `Intel(R) Xeon(R) Gold 6430`

### Experiment Boundary

- Do not rerun or overwrite the already completed baseline and warmup experiments from `2026-04-16`; keep them as historical results from the earlier hardware context.
- Any new measurements taken after `2026-04-21` must explicitly state that they were collected on the current `8 x RTX 4090` machine.

## 2026-04-23 Phase 14: Rebuild Runtime Environment On Current Machine

- Status: PASS
- Purpose: Rebuild a working assignment2 runtime environment on the current machine, while continuing to use local `cs336-basics` instead of `cs336-basics-standard`.

### Preconditions

- Verified assignment2 root `pyproject.toml` still points to:
  - `cs336-basics = { path = "./cs336-basics", editable = true }`
- Confirmed the old `llmPart2` environment was not usable on this machine because its `pyvenv.cfg` still referenced:
  - `/home/u-lidz/anaconda3/bin`

### Command

```bash
UV_PROJECT_ENVIRONMENT=llmPart2_runtime llmPart2/bin/uv sync --python /usr/bin/python3 --frozen
```

### Key Results

- Created a fresh runtime environment at:
  - `assignment2-systems/llmPart2_runtime`
- Synced `75` packages successfully
- Installed editable local packages:
  - `cs336-basics==1.0.3` from `./cs336-basics`
  - `cs336-systems==1.0.5` from current assignment2 root
- Installed key runtime packages:
  - `torch==2.6.0`
  - `triton==3.2.0`
  - `pytest==8.4.1`

### Validation

```bash
llmPart2_runtime/bin/python -c "import cs336_basics, torch, triton; print({'cs336_basics': cs336_basics.__file__, 'torch': torch.__version__, 'triton': triton.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count()})"
```

### Validation Result

- `cs336_basics` imports from:
  - `/home/lidz/cs336/code part/part2_systems/assignment2-systems/cs336-basics/cs336_basics/__init__.py`
- `torch == 2.6.0+cu124`
- `triton == 3.2.0`
- CUDA visible device count: `8`

## 2026-04-23 Phase 15: Extend `benchmark.py` For Mixed Precision And Memory Profiling

- Status: PASS
- Purpose: Add the code paths required for assignment `1.1.5 mixed precision` and `1.1.6 memory profiling`, without changing the existing baseline benchmark behavior.

### Code Changes

- Updated [benchmark.py](/home/lidz/cs336/code%20part/part2_systems/assignment2-systems/benchmark.py)

### New Capabilities

- Added `--mixed-precision {none,bf16}`
- Added CUDA autocast support for benchmark execution
- Added `--memory-profile`
- Added `--memory-history-max-entries`
- Added `--memory-snapshot-path`
- Added peak memory summary output:
  - `peak_memory_allocated_mb`
  - `peak_memory_reserved_mb`
- Added support for explicit device strings such as:
  - `cuda:0`
  - `cuda:7`

### Implementation Notes

- Mixed precision is enabled only through CUDA autocast.
- Model parameters remain in `torch.float32`; autocast changes op execution dtype where appropriate.
- Memory history recording begins after warmup and covers the measurement phase only.
- Snapshot dumping uses `torch.cuda.memory._dump_snapshot(...)`.

## 2026-04-23 Phase 16: Mixed Precision Accumulation Experiment

- Status: PASS
- Purpose: Execute the assignment `mixed_precision_accumulation` experiment and record the numerical behavior before the larger BF16 benchmark sweep.

### Command

```bash
llmPart2_runtime/bin/python - <<'PY'
import torch

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print('fp32_accum_fp32_input', float(s))

s = torch.tensor(0, dtype=torch.float16)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print('fp16_accum_fp16_input', float(s))

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print('fp32_accum_fp16_input', float(s))

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print('fp32_accum_casted_fp16_input', float(s))
PY
```

### Results

- `fp32_accum_fp32_input = 10.000133514404297`
- `fp16_accum_fp16_input = 9.953125`
- `fp32_accum_fp16_input = 10.00213623046875`
- `fp32_accum_casted_fp16_input = 10.00213623046875`

### Observation

- Low-precision accumulation is visibly less accurate than FP32 accumulation.
- Keeping the accumulator in FP32 preserves most of the accuracy, even when the inputs originate in FP16.

## 2026-04-23 Phase 17: GPU Resource Gate Before Formal Mixed Precision / Memory Runs

- Status: BLOCKED
- Purpose: Enforce a GPU availability check before each benchmark/profiling run and defer formal measurements if the machine is saturated.

### GPU Check Commands

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

### Observed State

- All `8` GPUs were occupied by another user's long-running Python workload.
- Representative state during the attempted `2026-04-23` session:
  - GPU memory used per card: roughly `21-22.5 GiB`
  - Free memory per card: roughly `1.5-3.1 GiB`
  - GPU utilization per card: roughly `92%-100%`
- Active compute processes were all from:
  - `/home/u-liujc/anaconda3/envs/dllm-rl/bin/python`

### Decision

- Postpone the formal `1.1.5(c)` BF16 benchmark sweep and the formal `1.1.6` memory profiling runs until GPUs are no longer saturated.
- Do not record timing or memory measurements collected under this contention as formal assignment results.

## 2026-04-27 Phase 18: Current Hardware And GPU Availability Check

- Status: PASS
- Purpose: Record the current hardware and GPU availability before resuming formal assignment2 benchmark/profiling work.
- Timestamp: `2026-04-27 15:29:31 CST +0800`

### Commands

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
lscpu
free -h
df -h '/home/lidz/cs336/code part/part2_systems/assignment2-systems'
```

### Key Results

- GPU: `8 x NVIDIA GeForce RTX 4090`
- NVIDIA driver: `590.48.01`
- GPU memory: `24564 MiB` per GPU
- Current GPU state:
  - Memory used: `15 MiB` per GPU
  - Free memory: `24069 MiB` per GPU
  - GPU utilization: `0%` on all 8 GPUs
  - Temperature range: `21-24 C`
- Active GPU compute processes: none reported by `nvidia-smi --query-compute-apps`
- CPU: `Intel(R) Xeon(R) Gold 6430`
- CPU topology:
  - `2` sockets
  - `32` cores per socket
  - `2` threads per core
  - `128` logical CPUs total
- NUMA nodes: `2`
- System memory:
  - Total: `1.0 Ti`
  - Used: `20 Gi`
  - Free: `711 Gi`
  - Available: `987 Gi`
- Swap:
  - Total: `8.0 Gi`
  - Used: `160 Ki`
- Assignment workspace filesystem:
  - Mount: `/home`
  - Device: `/dev/sdb1`
  - Size: `15T`
  - Used: `386G`
  - Available: `14T`
  - Use: `3%`

### Experiment Boundary

- Unlike the blocked `2026-04-23` check, the GPUs are currently available for formal benchmark and memory profiling runs.
- Future measurements taken from this point should state that they were collected on the `8 x RTX 4090`, `24564 MiB/GPU`, driver `590.48.01` machine.


## 2026-04-27 Formal Current-Hardware Benchmark Run `2026-04-27_rtx4090_formal`

- Status: PASS
- Purpose: Run the formal current-hardware benchmark sequence through baseline, warmup sweep, BF16 mixed precision, and memory profiling.
- Hardware context: `8 x NVIDIA GeForce RTX 4090`, `24564 MiB/GPU`, NVIDIA driver `590.48.01`.
- Output directory: `benchmark_outputs/2026-04-27_rtx4090_formal`
- Normalized report: `benchmark_outputs/2026-04-27_rtx4090_formal/RESULTS.md`
- CSV summary: `benchmark_outputs/2026-04-27_rtx4090_formal/summary.csv`
- Markdown summary: `benchmark_outputs/2026-04-27_rtx4090_formal/summary.md`

### Phase Summary

| phase | pass | oom | fail |
|---|---:|---:|---:|
| `01_baseline_fp32` | 9 | 1 | 0 |
| `02_warmup_sweep_fp32` | 36 | 4 | 0 |
| `03_mixed_precision_bf16` | 9 | 1 | 0 |
| `04_memory_profiling` | 6 | 6 | 0 |

### Notes

- OOM results are recorded as expected hardware-bound outcomes, not script failures.
- The run script keeps a per-GPU worker queue so available GPUs continue receiving work until the fourth phase completes.
- Use `RESULTS.md` as the human-readable report, `summary.csv` for machine-readable results, and the per-task logs for stderr, OOM traces, and raw benchmark output.
