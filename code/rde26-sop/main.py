#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


ALGORITHM = "rde26-sop"
TRACK = "SOP"
DIMENSION = 30
LOWER_BOUND = -100.0
UPPER_BOUND = 100.0
DEFAULT_MAX_EVALUATIONS = 300000
DEFAULT_RUNS = 25
DEFAULT_SEED_BASE = 20260601


def parse_id_range(text: str) -> list[int]:
    items: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
            start = int(left)
            end = int(right)
            step = 1 if start <= end else -1
            items.extend(range(start, end + step, step))
        else:
            items.append(int(part))
    return items


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run the RDE26-SOP algorithm.")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs" / f"{ALGORITHM}_{stamp}")
    parser.add_argument("--functions", default="1:29")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument("--paper-id", default="RDE26")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def build_binary(root: Path, cxx: str) -> Path:
    sandbox = root / "src" / "rdex_sandbox"
    build_dir = root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "rde26_sop"
    command = [
        cxx,
        "-O3",
        "-std=c++14",
        "rdex_sop_cli.cpp",
        "rdex_sop.cpp",
        "../evolutionary_computation/problems/cec2017_sop/cec17_func.cpp",
        "../evolutionary_computation/optimizers/cpp_bridge/problem_bridge.cpp",
        "-I.",
        "-I../evolutionary_computation/problems/cec2017_sop",
        "-I../evolutionary_computation/optimizers/cpp_bridge",
        "-o",
        str(binary),
    ]
    run_command(command, sandbox)
    return binary


def normalize_trace(values: list[float], rows: int) -> list[float]:
    if not values:
        raise RuntimeError("empty trace")
    if len(values) >= rows:
        return values[:rows]
    return values + [values[-1]] * (rows - len(values))


def write_matrix(path: Path, columns: list[list[float]]) -> None:
    if not columns:
        raise RuntimeError(f"no data for {path.name}")
    row_count = len(columns[0])
    if any(len(col) != row_count for col in columns):
        raise RuntimeError(f"inconsistent trace length for {path.name}")
    with path.open("w", encoding="utf-8") as handle:
        for row in range(row_count):
            handle.write("\t".join(f"{col[row]:.17g}" for col in columns))
            handle.write("\n")


def run_algorithm(binary: Path, cec_dir: Path, function_id: int, seed: int, max_evaluations: int) -> list[float]:
    completed = subprocess.run(
        [
            str(binary),
            str(function_id),
            str(seed),
            str(max_evaluations),
            str(DIMENSION),
            str(LOWER_BOUND),
            str(UPPER_BOUND),
            str(cec_dir),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    values: list[float] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            values.append(float(parts[1]))
    return values


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.max_evaluations < 1:
        raise SystemExit("--max-evaluations must be positive")

    root = Path(__file__).resolve().parent
    binary = root / "build" / "rde26_sop"
    if not args.skip_build or not binary.is_file():
        binary = build_binary(root, args.cxx)
    cec_dir = root / "src" / "evolutionary_computation" / "problems" / "cec2017_sop"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    functions = parse_id_range(args.functions)
    rows = max(1, args.max_evaluations // (10 * DIMENSION))

    for function_id in functions:
        if function_id < 1 or function_id > 29:
            raise SystemExit(f"SOP function id out of range: {function_id}")
        run_columns: list[list[float]] = []
        for run_id in range(1, args.runs + 1):
            seed = args.seed_base + function_id * 1000 + run_id
            trace = run_algorithm(binary, cec_dir, function_id, seed, int(args.max_evaluations))
            run_columns.append(normalize_trace(trace, rows))
        target = args.out_dir / f"{args.paper_id}_CEC26_{TRACK}_F{function_id}.txt"
        write_matrix(target, run_columns)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
