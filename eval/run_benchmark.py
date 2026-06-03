"""Full evaluation benchmark on LEVIR-CD test set."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="levir_cd")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="eval/results/benchmark_final.json")
    args = parser.parse_args()

    raise NotImplementedError("Phase 10")


if __name__ == "__main__":
    main()
