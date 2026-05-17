from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from time import monotonic


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark.py"
EXPERIMENT_LOG = ROOT / "assignment2_experiment_log.md"

MODEL_SIZES = ("small", "medium", "large", "xl", "2.7b")
MODES = ("forward", "forward_backward")
WARMUP_VALUES = (0, 1, 2, 5)
MEMORY_CONTEXT_LENGTHS = (128, 256, 512)
MEMORY_MODES = ("forward", "train_step")
MEMORY_PRECISIONS = ("none", "bf16")

SUMMARY_PATTERN = re.compile(
    r"summary\n"
    r"  mean_ms=(?P<mean_ms>[0-9.]+)\n"
    r"  std_ms=(?P<std_ms>[0-9.]+)\n"
    r"  min_ms=(?P<min_ms>[0-9.]+)\n"
    r"  max_ms=(?P<max_ms>[0-9.]+)\n"
    r"  tokens_per_step=(?P<tokens_per_step>[0-9]+)\n"
    r"  tokens_per_second=(?P<tokens_per_second>[0-9.]+)"
    r"(?:\n  peak_memory_allocated_mb=(?P<peak_memory_allocated_mb>[0-9.]+))?"
    r"(?:\n  peak_memory_reserved_mb=(?P<peak_memory_reserved_mb>[0-9.]+))?"
)


@dataclass(frozen=True)
class Task:
    phase: str
    name: str
    model_size: str
    mode: str
    context_length: int
    warmup_steps: int
    measurement_steps: int
    mixed_precision: str
    memory_profile: bool
    log_path: str
    snapshot_path: str | None


@dataclass
class TaskResult:
    phase: str
    name: str
    gpu: int
    status: str
    returncode: int
    elapsed_seconds: float
    log_path: str
    snapshot_path: str | None
    model_size: str
    mode: str
    context_length: int
    warmup_steps: int
    measurement_steps: int
    mixed_precision: str
    memory_profile: bool
    mean_ms: float | None = None
    std_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    tokens_per_step: float | None = None
    tokens_per_second: float | None = None
    peak_memory_allocated_mb: float | None = None
    peak_memory_reserved_mb: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current-hardware CS336 assignment2 benchmark phases with "
            "multi-GPU scheduling, summaries, and experiment-log append."
        )
    )
    parser.add_argument("--run-name", default="2026-04-27_rtx4090_formal")
    parser.add_argument("--python", default=None, help="Python interpreter to use for benchmark.py.")
    parser.add_argument("--max-gpus", type=int, default=None, help="Limit the number of GPUs used.")
    parser.add_argument(
        "--required-gpus",
        type=int,
        default=8,
        help="Require this many GPUs before starting. Use 0 to disable the check.",
    )
    parser.add_argument(
        "--strict-phase-order",
        action="store_true",
        help=(
            "Finish each phase before starting the next. By default, all phases are placed "
            "in one FIFO queue so GPUs keep receiving work even when a phase tail has fewer "
            "remaining tasks than available GPUs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--baseline-warmup-steps", type=int, default=5)
    parser.add_argument("--baseline-measurement-steps", type=int, default=10)
    parser.add_argument("--memory-warmup-steps", type=int, default=1)
    parser.add_argument("--memory-measurement-steps", type=int, default=1)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-warmup-sweep", action="store_true")
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only write the task manifest; do not run benchmarks.")
    return parser.parse_args()


def resolve_python(explicit_python: str | None) -> str:
    if explicit_python:
        return explicit_python
    runtime_python = ROOT / "llmPart2_runtime" / "bin" / "python"
    if runtime_python.exists():
        return str(runtime_python)
    return sys.executable


def detect_gpus(max_gpus: int | None, required_gpus: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed:\n{completed.stderr}")

    gpus = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if max_gpus is not None:
        gpus = gpus[:max_gpus]
    if not gpus:
        raise RuntimeError("No GPUs detected.")
    if required_gpus > 0 and len(gpus) < required_gpus:
        raise RuntimeError(
            f"Detected {len(gpus)} GPU(s), but this run requires {required_gpus}. "
            "Do not pass --max-gpus for the formal all-card run."
        )
    return gpus


def slug(*parts: object) -> str:
    return "_".join(str(part).replace(".", "p").replace("-", "_") for part in parts)


def make_task(
    output_dir: Path,
    phase: str,
    model_size: str,
    mode: str,
    context_length: int,
    warmup_steps: int,
    measurement_steps: int,
    mixed_precision: str,
    memory_profile: bool = False,
) -> Task:
    name = slug(
        phase,
        "size",
        model_size,
        "mode",
        mode,
        "ctx",
        context_length,
        "warmup",
        warmup_steps,
        "mp",
        mixed_precision,
    )
    log_path = output_dir / "logs" / f"{name}.log"
    snapshot_path = None
    if memory_profile:
        snapshot_path = str(output_dir / "memory_snapshots" / f"{name}.pickle")
    return Task(
        phase=phase,
        name=name,
        model_size=model_size,
        mode=mode,
        context_length=context_length,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
        mixed_precision=mixed_precision,
        memory_profile=memory_profile,
        log_path=str(log_path),
        snapshot_path=snapshot_path,
    )


def build_tasks(args: argparse.Namespace, output_dir: Path) -> list[Task]:
    tasks: list[Task] = []

    if not args.skip_baseline:
        for model_size in MODEL_SIZES:
            for mode in MODES:
                tasks.append(
                    make_task(
                        output_dir,
                        "01_baseline_fp32",
                        model_size,
                        mode,
                        args.context_length,
                        args.baseline_warmup_steps,
                        args.baseline_measurement_steps,
                        "none",
                    )
                )

    if not args.skip_warmup_sweep:
        for model_size in MODEL_SIZES:
            for mode in MODES:
                for warmup_steps in WARMUP_VALUES:
                    tasks.append(
                        make_task(
                            output_dir,
                            "02_warmup_sweep_fp32",
                            model_size,
                            mode,
                            args.context_length,
                            warmup_steps,
                            args.baseline_measurement_steps,
                            "none",
                        )
                    )

    if not args.skip_bf16:
        for model_size in MODEL_SIZES:
            for mode in MODES:
                tasks.append(
                    make_task(
                        output_dir,
                        "03_mixed_precision_bf16",
                        model_size,
                        mode,
                        args.context_length,
                        args.baseline_warmup_steps,
                        args.baseline_measurement_steps,
                        "bf16",
                    )
                )

    if not args.skip_memory:
        for context_length in MEMORY_CONTEXT_LENGTHS:
            for mixed_precision in MEMORY_PRECISIONS:
                for mode in MEMORY_MODES:
                    tasks.append(
                        make_task(
                            output_dir,
                            "04_memory_profiling",
                            "2.7b",
                            mode,
                            context_length,
                            args.memory_warmup_steps,
                            args.memory_measurement_steps,
                            mixed_precision,
                            memory_profile=True,
                        )
                    )

    return tasks


def command_for_task(python_executable: str, task: Task, args: argparse.Namespace) -> list[str]:
    command = [
        python_executable,
        str(BENCHMARK),
        "--model-size",
        task.model_size,
        "--mode",
        task.mode,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(task.context_length),
        "--vocab-size",
        str(args.vocab_size),
        "--warmup-steps",
        str(task.warmup_steps),
        "--measurement-steps",
        str(task.measurement_steps),
        "--device",
        "cuda:0",
        "--mixed-precision",
        task.mixed_precision,
    ]
    if task.memory_profile:
        command.extend(
            [
                "--memory-profile",
                "--memory-snapshot-path",
                str(task.snapshot_path),
            ]
        )
    return command


def parse_result(task: Task, gpu: int, returncode: int, elapsed_seconds: float, output_text: str) -> TaskResult:
    lower_text = output_text.lower()
    if returncode == 0:
        status = "PASS"
    elif "outofmemoryerror" in lower_text or "cuda out of memory" in lower_text:
        status = "OOM"
    else:
        status = "FAIL"

    result = TaskResult(
        phase=task.phase,
        name=task.name,
        gpu=gpu,
        status=status,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        log_path=task.log_path,
        snapshot_path=task.snapshot_path,
        model_size=task.model_size,
        mode=task.mode,
        context_length=task.context_length,
        warmup_steps=task.warmup_steps,
        measurement_steps=task.measurement_steps,
        mixed_precision=task.mixed_precision,
        memory_profile=task.memory_profile,
    )

    match = SUMMARY_PATTERN.search(output_text)
    if match is None:
        return result

    for field in (
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "tokens_per_step",
        "tokens_per_second",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
    ):
        value = match.group(field)
        if value is not None:
            setattr(result, field, float(value))
    return result


def run_phase(
    phase: str,
    tasks: list[Task],
    gpus: list[int],
    python_executable: str,
    args: argparse.Namespace,
) -> list[TaskResult]:
    if not tasks:
        return []

    queue: Queue[Task] = Queue()
    for task in tasks:
        queue.put(task)

    results: list[TaskResult] = []
    results_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                task = queue.get_nowait()
            except Exception:
                return

            log_path = Path(task.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if task.snapshot_path is not None:
                Path(task.snapshot_path).parent.mkdir(parents=True, exist_ok=True)

            command = command_for_task(python_executable, task, args)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("PYTHONUNBUFFERED", "1")

            start = monotonic()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )
            elapsed_seconds = monotonic() - start

            output_text = completed.stdout
            if completed.stderr:
                output_text += "\n[stderr]\n" + completed.stderr

            header = [
                f"phase={phase}",
                f"task={task.name}",
                f"gpu={gpu}",
                f"command={' '.join(command)}",
                f"start_elapsed_marker_seconds={start:.6f}",
                "",
            ]
            log_path.write_text("\n".join(header) + output_text)

            result = parse_result(task, gpu, completed.returncode, elapsed_seconds, output_text)
            with results_lock:
                results.append(result)
                print(
                    f"[{phase}] gpu={gpu} status={result.status} "
                    f"task={task.name} elapsed_s={elapsed_seconds:.1f}",
                    flush=True,
                )
            queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return sorted(results, key=lambda result: result.name)


def write_summary(output_dir: Path, results: list[TaskResult]) -> tuple[Path, Path]:
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"

    fields = list(asdict(results[0]).keys()) if results else list(TaskResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    with md_path.open("w") as f:
        f.write("| phase | status | model | mode | ctx | warmup | mp | mean_ms | std_ms | peak_alloc_mb | log |\n")
        f.write("|---|---|---:|---|---:|---:|---|---:|---:|---:|---|\n")
        for result in results:
            f.write(
                "| "
                f"{result.phase} | {result.status} | {result.model_size} | {result.mode} | "
                f"{result.context_length} | {result.warmup_steps} | {result.mixed_precision} | "
                f"{format_optional(result.mean_ms)} | {format_optional(result.std_ms)} | "
                f"{format_optional(result.peak_memory_allocated_mb)} | {Path(result.log_path).name} |\n"
            )

    return csv_path, md_path


def format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def append_experiment_log(run_name: str, output_dir: Path, results: list[TaskResult], csv_path: Path, md_path: Path) -> None:
    by_phase: dict[str, list[TaskResult]] = {}
    for result in results:
        by_phase.setdefault(result.phase, []).append(result)

    lines = [
        "",
        f"## {datetime.now().strftime('%Y-%m-%d')} Formal Current-Hardware Benchmark Run `{run_name}`",
        "",
        "- Status: PASS" if all(result.status in {"PASS", "OOM"} for result in results) else "- Status: REVIEW",
        "- Purpose: Run the formal current-hardware benchmark sequence through baseline, warmup sweep, BF16 mixed precision, and memory profiling.",
        "- Hardware context: `8 x NVIDIA GeForce RTX 4090`, `24564 MiB/GPU`, NVIDIA driver `590.48.01`.",
        f"- Output directory: `{output_dir.relative_to(ROOT)}`",
        f"- CSV summary: `{csv_path.relative_to(ROOT)}`",
        f"- Markdown summary: `{md_path.relative_to(ROOT)}`",
        "",
        "### Phase Summary",
        "",
        "| phase | pass | oom | fail |",
        "|---|---:|---:|---:|",
    ]

    for phase in sorted(by_phase):
        phase_results = by_phase[phase]
        pass_count = sum(result.status == "PASS" for result in phase_results)
        oom_count = sum(result.status == "OOM" for result in phase_results)
        fail_count = sum(result.status == "FAIL" for result in phase_results)
        lines.append(f"| `{phase}` | {pass_count} | {oom_count} | {fail_count} |")

    lines.extend(
        [
            "",
            "### Notes",
            "",
            "- OOM results are recorded as expected hardware-bound outcomes, not script failures.",
            "- The run script keeps a per-GPU worker queue so available GPUs continue receiving work until the fourth phase completes.",
            "- Use the per-task logs for stderr, OOM traces, and raw benchmark output.",
        ]
    )
    EXPERIMENT_LOG.write_text(EXPERIMENT_LOG.read_text() + "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    python_executable = resolve_python(args.python)
    gpus = detect_gpus(args.max_gpus, args.required_gpus)

    output_dir = ROOT / "benchmark_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, output_dir)
    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": python_executable,
        "gpus": gpus,
        "args": vars(args),
        "tasks": [asdict(task) for task in tasks],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"run_name={args.run_name}")
    print(f"python={python_executable}")
    print(f"gpus={','.join(str(gpu) for gpu in gpus)}")
    print(f"task_count={len(tasks)}")
    print(f"output_dir={output_dir}")

    if args.dry_run:
        print("dry_run=true")
        return

    all_results: list[TaskResult] = []
    if args.strict_phase_order:
        for phase in ("01_baseline_fp32", "02_warmup_sweep_fp32", "03_mixed_precision_bf16", "04_memory_profiling"):
            phase_tasks = [task for task in tasks if task.phase == phase]
            print(f"starting_phase={phase} tasks={len(phase_tasks)}")
            phase_results = run_phase(phase, phase_tasks, gpus, python_executable, args)
            all_results.extend(phase_results)
            csv_path, md_path = write_summary(output_dir, all_results)
            print(f"completed_phase={phase} cumulative_summary={csv_path}")
    else:
        print(f"starting_global_queue tasks={len(tasks)}")
        all_results = run_phase("all_phases_continuous", tasks, gpus, python_executable, args)
        csv_path, md_path = write_summary(output_dir, all_results)
        print(f"completed_global_queue cumulative_summary={csv_path}")

    csv_path, md_path = write_summary(output_dir, all_results)
    append_experiment_log(args.run_name, output_dir, all_results, csv_path, md_path)
    print(f"final_csv={csv_path}")
    print(f"final_markdown={md_path}")
    print("completed_all_phases=true")


if __name__ == "__main__":
    main()
