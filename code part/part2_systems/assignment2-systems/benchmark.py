from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer

import torch

from cs336_basics import BasicsTransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.cross_entropy import cross_entropy as cross_entropy_fn


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

SUMMARY_PATTERN = re.compile(
    r"summary\n"
    r"  mean_ms=(?P<mean_ms>[0-9.]+)\n"
    r"  std_ms=(?P<std_ms>[0-9.]+)\n"
    r"  min_ms=(?P<min_ms>[0-9.]+)\n"
    r"  max_ms=(?P<max_ms>[0-9.]+)\n"
    r"  tokens_per_step=(?P<tokens_per_step>[0-9]+)\n"
    r"  tokens_per_second=(?P<tokens_per_second>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the custom CS336 Transformer LM.")
    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS,
        default="small",
        help="Preset transformer size to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        default="train_step",
        help="Workload to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device. 'auto' selects CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup-sweep",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Run one benchmark per listed warmup count in a fresh subprocess and "
            "print an aggregate summary."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[args.model_size]
    return BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (args.batch_size, args.context_length)
    inputs = torch.randint(0, args.vocab_size, shape, device=device, dtype=torch.long)
    targets = torch.randint(0, args.vocab_size, shape, device=device, dtype=torch.long)
    return inputs, targets


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    return cross_entropy_fn(flat_logits, flat_targets)


def run_forward(model: BasicsTransformerLM, inputs: torch.Tensor) -> None:
    with torch.no_grad():
        model(inputs)


def run_forward_backward(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = compute_loss(logits, targets)
    loss.backward()


def run_train_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = compute_loss(logits, targets)
    loss.backward()
    optimizer.step()


def benchmark_step(
    mode: str,
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if mode == "forward":
        run_forward(model, inputs)
        return
    if mode == "forward_backward":
        run_forward_backward(model, inputs, targets)
        return
    if optimizer is None:
        raise RuntimeError("train_step mode requires an optimizer.")
    run_train_step(model, optimizer, inputs, targets)


def summarize_times(times: list[float]) -> dict[str, float]:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "mean_ms": mean * 1000.0,
        "std_ms": stdev * 1000.0,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
    }


def print_run_header(
    args: argparse.Namespace,
    device: torch.device,
    model: BasicsTransformerLM,
    warmup_steps: int,
) -> None:
    model.train(args.mode != "forward")
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Mode: {args.mode}")
    print(f"Model size preset: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Parameter count: {model.get_num_params(non_embedding=False)}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Measurement steps: {args.measurement_steps}")


def run_benchmark(
    args: argparse.Namespace,
    warmup_steps: int | None = None,
) -> dict[str, float]:
    if args.measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive.")

    effective_warmup_steps = args.warmup_steps if warmup_steps is None else warmup_steps
    if effective_warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = make_model(args, device)
    model.train(args.mode != "forward")
    optimizer = None
    if args.mode == "train_step":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    inputs, targets = make_batch(args, device)

    print_run_header(args, device, model, effective_warmup_steps)

    for step_idx in range(effective_warmup_steps):
        maybe_synchronize(device)
        start = default_timer()
        benchmark_step(args.mode, model, optimizer, inputs, targets)
        maybe_synchronize(device)
        elapsed = default_timer() - start
        print(f"warmup_step={step_idx + 1} time_ms={elapsed * 1000.0:.3f}")

    measurement_times: list[float] = []
    for step_idx in range(args.measurement_steps):
        maybe_synchronize(device)
        start = default_timer()
        benchmark_step(args.mode, model, optimizer, inputs, targets)
        maybe_synchronize(device)
        elapsed = default_timer() - start
        measurement_times.append(elapsed)
        print(f"measurement_step={step_idx + 1} time_ms={elapsed * 1000.0:.3f}")

    summary = summarize_times(measurement_times)
    tokens_per_step = args.batch_size * args.context_length
    tokens_per_second = tokens_per_step / (summary["mean_ms"] / 1000.0)

    print("summary")
    print(f"  mean_ms={summary['mean_ms']:.3f}")
    print(f"  std_ms={summary['std_ms']:.3f}")
    print(f"  min_ms={summary['min_ms']:.3f}")
    print(f"  max_ms={summary['max_ms']:.3f}")
    print(f"  tokens_per_step={tokens_per_step}")
    print(f"  tokens_per_second={tokens_per_second:.3f}")
    return {
        "warmup_steps": float(effective_warmup_steps),
        "mean_ms": summary["mean_ms"],
        "std_ms": summary["std_ms"],
        "min_ms": summary["min_ms"],
        "max_ms": summary["max_ms"],
        "tokens_per_step": float(tokens_per_step),
        "tokens_per_second": tokens_per_second,
    }


def build_subprocess_command(args: argparse.Namespace, warmup_steps: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-size",
        args.model_size,
        "--mode",
        args.mode,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--vocab-size",
        str(args.vocab_size),
        "--rope-theta",
        str(args.rope_theta),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]


def parse_summary(stdout: str, warmup_steps: int) -> dict[str, float]:
    match = SUMMARY_PATTERN.search(stdout)
    if match is None:
        raise RuntimeError(
            f"Failed to parse benchmark summary for warmup_steps={warmup_steps}."
        )
    return {
        "warmup_steps": float(warmup_steps),
        "mean_ms": float(match.group("mean_ms")),
        "std_ms": float(match.group("std_ms")),
        "min_ms": float(match.group("min_ms")),
        "max_ms": float(match.group("max_ms")),
        "tokens_per_step": float(match.group("tokens_per_step")),
        "tokens_per_second": float(match.group("tokens_per_second")),
    }


def run_warmup_sweep(args: argparse.Namespace) -> None:
    if args.warmup_sweep is None:
        raise RuntimeError("warmup_sweep was requested but no values were provided.")
    if any(value < 0 for value in args.warmup_sweep):
        raise ValueError("warmup_sweep values must all be non-negative.")

    print("warmup_sweep_config")
    print(f"  mode={args.mode}")
    print(f"  model_size={args.model_size}")
    print(f"  batch_size={args.batch_size}")
    print(f"  context_length={args.context_length}")
    print(f"  vocab_size={args.vocab_size}")
    print(f"  measurement_steps={args.measurement_steps}")
    print(f"  device={args.device}")
    print(f"  sweep_values={','.join(str(value) for value in args.warmup_sweep)}")

    results: list[dict[str, float]] = []
    script_dir = Path(__file__).resolve().parent
    for warmup_steps in args.warmup_sweep:
        print(f"=== warmup_sweep warmup_steps={warmup_steps} ===")
        completed = subprocess.run(
            build_subprocess_command(args, warmup_steps),
            cwd=script_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode != 0:
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            raise RuntimeError(
                f"Warmup sweep child benchmark failed for warmup_steps={warmup_steps}."
            )
        results.append(parse_summary(completed.stdout, warmup_steps))

    print("warmup_sweep_summary")
    for result in results:
        print(
            "  "
            f"warmup_steps={int(result['warmup_steps'])} "
            f"mean_ms={result['mean_ms']:.3f} "
            f"std_ms={result['std_ms']:.3f} "
            f"min_ms={result['min_ms']:.3f} "
            f"max_ms={result['max_ms']:.3f} "
            f"tokens_per_second={result['tokens_per_second']:.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.warmup_sweep is not None:
        run_warmup_sweep(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
