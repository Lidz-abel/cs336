from __future__ import annotations

import argparse
import csv
import math
import statistics
import timeit
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive PyTorch attention for CS336 assignment2.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-values", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--implementations",
        choices=["vanilla", "compiled"],
        nargs="+",
        default=["vanilla", "compiled"],
        help="Attention implementations to benchmark.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode used for the compiled implementation.",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="torch.compile backend used for the compiled implementation.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/2026-05-06_rtx4090_attention_baseline",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def make_attention_impl(name: str, compile_mode: str, compile_backend: str):
    if name == "vanilla":
        return attention
    if name == "compiled":
        kwargs = {} if compile_mode == "default" else {"mode": compile_mode}
        kwargs["backend"] = compile_backend
        return torch.compile(attention, **kwargs)
    raise ValueError(f"Unsupported attention implementation: {name}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def allocated_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**2)


def summarize(times: list[float]) -> tuple[float, float]:
    mean_ms = statistics.mean(times) * 1000.0
    std_ms = statistics.stdev(times) * 1000.0 if len(times) > 1 else 0.0
    return mean_ms, std_ms


def make_inputs(
    batch_size: int,
    seq_len: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def run_forward_benchmark(
    attention_impl,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    measurement_steps: int,
) -> tuple[str, float | None, float | None, float | None, str]:
    reset_memory(device)
    try:
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = attention_impl(q, k, v)
                sync(device)

            times = []
            for _ in range(measurement_steps):
                start = timeit.default_timer()
                _ = attention_impl(q, k, v)
                sync(device)
                times.append(timeit.default_timer() - start)

        mean_ms, std_ms = summarize(times)
        return "PASS", mean_ms, std_ms, peak_memory_mb(device), ""
    except torch.cuda.OutOfMemoryError as exc:
        reset_memory(device)
        return "OOM", None, None, None, normalize_error(exc)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            reset_memory(device)
            return "OOM", None, None, None, normalize_error(exc)
        return "FAIL", None, None, None, normalize_error(exc)


def run_backward_benchmark(
    attention_impl,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    measurement_steps: int,
) -> tuple[str, float | None, float | None, float | None, float | None, str]:
    reset_memory(device)
    peak_before_backward_mb = 0.0

    try:
        for _ in range(warmup_steps):
            q.grad = None
            k.grad = None
            v.grad = None
            out = attention_impl(q, k, v)
            grad_out = torch.randn_like(out)
            peak_before_backward_mb = max(peak_before_backward_mb, allocated_memory_mb(device))
            out.backward(grad_out)
            sync(device)

        times = []
        for _ in range(measurement_steps):
            q.grad = None
            k.grad = None
            v.grad = None
            out = attention_impl(q, k, v)
            grad_out = torch.randn_like(out)
            peak_before_backward_mb = max(peak_before_backward_mb, allocated_memory_mb(device))

            start = timeit.default_timer()
            out.backward(grad_out)
            sync(device)
            times.append(timeit.default_timer() - start)

        mean_ms, std_ms = summarize(times)
        return "PASS", mean_ms, std_ms, peak_before_backward_mb, peak_memory_mb(device), ""
    except torch.cuda.OutOfMemoryError as exc:
        reset_memory(device)
        return "OOM", None, None, peak_before_backward_mb or None, None, normalize_error(exc)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            reset_memory(device)
            return "OOM", None, None, peak_before_backward_mb or None, None, normalize_error(exc)
        return "FAIL", None, None, peak_before_backward_mb or None, None, normalize_error(exc)


def normalize_error(exc: BaseException) -> str:
    return str(exc).replace("\n", " ")[:1000]


def theoretical_score_matrix_mib(batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    return batch_size * seq_len * seq_len * bytes_per_element / (1024**2)


def theoretical_qkv_mib(batch_size: int, seq_len: int, d: int, dtype: torch.dtype) -> float:
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    return 3 * batch_size * seq_len * d * bytes_per_element / (1024**2)


def fmt(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "| impl | batch | seq_len | d | dtype | fwd status | fwd mean ms | fwd std ms | bwd status | bwd mean ms | bwd std ms | score matrix MiB | peak fwd MB | peak bwd MB |",
        "|---|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['implementation']} | "
            f"{row['batch_size']} | "
            f"{row['seq_len']} | "
            f"{row['d']} | "
            f"{row['dtype']} | "
            f"{row['forward_status']} | "
            f"{fmt(row['forward_mean_ms'])} | "
            f"{fmt(row['forward_std_ms'])} | "
            f"{row['backward_status']} | "
            f"{fmt(row['backward_mean_ms'])} | "
            f"{fmt(row['backward_std_ms'])} | "
            f"{fmt(row['score_matrix_mib'])} | "
            f"{fmt(row['peak_forward_mb'])} | "
            f"{fmt(row['peak_backward_mb'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    forward_oom = [row for row in rows if row["forward_status"] == "OOM"]
    backward_oom = [row for row in rows if row["backward_status"] == "OOM"]
    forward_pass = [row for row in rows if row["forward_status"] == "PASS"]
    backward_pass = [row for row in rows if row["backward_status"] == "PASS"]

    lines = [
        "# PyTorch Attention Baseline And torch.compile Comparison",
        "",
        "This file summarizes the handout `pytorch_attention` and `torch_compile` attention sweeps.",
        "",
        "## Setup",
        "",
        "- Vanilla implementation: naive PyTorch attention using `matmul -> softmax -> matmul`.",
        "- Compiled implementation: the same Python function wrapped by `torch.compile`.",
        "- Shape: `Q, K, V = (batch_size, seq_len, d)`.",
        "- No multi-head dimension.",
        "- No causal mask.",
        "- Forward timing uses `torch.no_grad()`.",
        "- Backward timing measures `out.backward(grad_out)` after rebuilding the forward graph.",
        "",
        "## Outputs",
        "",
        "- Machine-readable summary: `summary.csv`",
        "- Markdown table: `summary.md`",
        "",
        "## Completion Summary",
        "",
        f"- Forward PASS/OOM/FAIL: `{count_status(rows, 'forward_status', 'PASS')}` / `{count_status(rows, 'forward_status', 'OOM')}` / `{count_status(rows, 'forward_status', 'FAIL')}`",
        f"- Backward PASS/OOM/FAIL: `{count_status(rows, 'backward_status', 'PASS')}` / `{count_status(rows, 'backward_status', 'OOM')}` / `{count_status(rows, 'backward_status', 'FAIL')}`",
        "",
        "## OOM Summary",
        "",
    ]

    if not forward_oom and not backward_oom:
        lines.append("- No OOM encountered.")
    else:
        for row in forward_oom:
            lines.append(
                f"- Forward OOM for `{row['implementation']}` at `seq_len={row['seq_len']}`, `d={row['d']}`, "
                f"one score matrix `{fmt(row['score_matrix_mib'])} MiB`."
            )
        for row in backward_oom:
            lines.append(
                f"- Backward OOM for `{row['implementation']}` at `seq_len={row['seq_len']}`, `d={row['d']}`, "
                f"one score matrix `{fmt(row['score_matrix_mib'])} MiB`."
            )

    lines.extend(["", "## Fastest And Slowest Successful Runs", ""])
    if forward_pass:
        fastest_forward = min(forward_pass, key=lambda row: float(row["forward_mean_ms"]))
        slowest_forward = max(forward_pass, key=lambda row: float(row["forward_mean_ms"]))
        lines.append(
            f"- Fastest forward: `seq_len={fastest_forward['seq_len']}`, `d={fastest_forward['d']}`, "
            f"`implementation={fastest_forward['implementation']}`, `{fmt(fastest_forward['forward_mean_ms'])} ms`."
        )
        lines.append(
            f"- Slowest forward: `seq_len={slowest_forward['seq_len']}`, `d={slowest_forward['d']}`, "
            f"`implementation={slowest_forward['implementation']}`, `{fmt(slowest_forward['forward_mean_ms'])} ms`."
        )
    if backward_pass:
        fastest_backward = min(backward_pass, key=lambda row: float(row["backward_mean_ms"]))
        slowest_backward = max(backward_pass, key=lambda row: float(row["backward_mean_ms"]))
        lines.append(
            f"- Fastest backward: `seq_len={fastest_backward['seq_len']}`, `d={fastest_backward['d']}`, "
            f"`implementation={fastest_backward['implementation']}`, `{fmt(fastest_backward['backward_mean_ms'])} ms`."
        )
        lines.append(
            f"- Slowest backward: `seq_len={slowest_backward['seq_len']}`, `d={slowest_backward['d']}`, "
            f"`implementation={slowest_backward['implementation']}`, `{fmt(slowest_backward['backward_mean_ms'])} ms`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Naive attention materializes score/probability tensors with shape "
            "`(batch_size, seq_len, seq_len)`, so memory grows quadratically with sequence length. "
            "For FP32, one score matrix at `batch=8, seq_len=16384` is about `8192 MiB`; backward "
            "requires additional saved tensors and gradients, which explains the OOM boundary. "
            "This is the baseline that FlashAttention is meant to improve by avoiding full materialization "
            "of the attention matrix.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def count_status(rows: list[dict[str, object]], key: str, status: str) -> int:
    return sum(row[key] == status for row in rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    torch.manual_seed(0)

    implementations = {
        implementation: make_attention_impl(implementation, args.compile_mode, args.compile_backend)
        for implementation in args.implementations
    }

    for implementation, attention_impl in implementations.items():
        for seq_len in args.seq_lens:
            for d in args.d_values:
                print(f"benchmark implementation={implementation} seq_len={seq_len} d={d} dtype={args.dtype}", flush=True)
                row: dict[str, object] = {
                    "implementation": implementation,
                    "compile_mode": args.compile_mode if implementation == "compiled" else "",
                    "compile_backend": args.compile_backend if implementation == "compiled" else "",
                    "batch_size": args.batch_size,
                    "seq_len": seq_len,
                    "d": d,
                    "dtype": args.dtype,
                    "forward_status": "PENDING",
                    "forward_mean_ms": None,
                    "forward_std_ms": None,
                    "backward_status": "PENDING",
                    "backward_mean_ms": None,
                    "backward_std_ms": None,
                    "peak_allocated_before_backward_mb": None,
                    "peak_forward_mb": None,
                    "peak_backward_mb": None,
                    "score_matrix_mib": theoretical_score_matrix_mib(args.batch_size, seq_len, dtype),
                    "qkv_mib": theoretical_qkv_mib(args.batch_size, seq_len, d, dtype),
                    "forward_error": "",
                    "backward_error": "",
                }

                try:
                    q, k, v = make_inputs(args.batch_size, seq_len, d, device, dtype)
                except torch.cuda.OutOfMemoryError as exc:
                    row["forward_status"] = "OOM"
                    row["backward_status"] = "OOM"
                    row["forward_error"] = normalize_error(exc)
                    row["backward_error"] = "Input allocation failed."
                    reset_memory(device)
                    rows.append(row)
                    continue

                (
                    row["forward_status"],
                    row["forward_mean_ms"],
                    row["forward_std_ms"],
                    row["peak_forward_mb"],
                    row["forward_error"],
                ) = run_forward_benchmark(
                    attention_impl,
                    q,
                    k,
                    v,
                    device,
                    args.warmup_steps,
                    args.measurement_steps,
                )

                (
                    row["backward_status"],
                    row["backward_mean_ms"],
                    row["backward_std_ms"],
                    row["peak_allocated_before_backward_mb"],
                    row["peak_backward_mb"],
                    row["backward_error"],
                ) = run_backward_benchmark(
                    attention_impl,
                    q,
                    k,
                    v,
                    device,
                    args.warmup_steps,
                    args.measurement_steps,
                )

                del q, k, v
                reset_memory(device)
                rows.append(row)

    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    results_path = output_dir / "RESULTS.md"

    fieldnames = [
        "implementation",
        "compile_mode",
        "compile_backend",
        "batch_size",
        "seq_len",
        "d",
        "dtype",
        "forward_status",
        "forward_mean_ms",
        "forward_std_ms",
        "backward_status",
        "backward_mean_ms",
        "backward_std_ms",
        "peak_allocated_before_backward_mb",
        "peak_forward_mb",
        "peak_backward_mb",
        "score_matrix_mib",
        "qkv_mib",
        "forward_error",
        "backward_error",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, md_path)
    write_results(rows, results_path)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
