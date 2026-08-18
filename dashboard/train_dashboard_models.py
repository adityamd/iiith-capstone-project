from __future__ import annotations

import argparse
import json
from pathlib import Path

# Load PyTorch before pandas/scikit-learn native libraries in the Windows MLDL environment.
import dashboard.torch_backend  # noqa: F401
from dashboard.modeling import benchmark_and_export, find_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the three-model dashboard benchmark.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve() if args.data_dir else find_data_dir(Path.cwd())
    print(f"Using OULAD data: {data_dir}")
    print(f"Writing artifacts: {args.artifact_dir.resolve()}")
    summary = benchmark_and_export(data_dir, args.artifact_dir.resolve())
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
