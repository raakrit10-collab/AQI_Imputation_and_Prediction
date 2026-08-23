import argparse
import sys
from src.pipeline.train_imputation import build_arg_parser as build_impute_parser, run_imputation
from src.pipeline.eval_imputation import build_arg_parser as build_eval_parser, run_evaluation
from src.pipeline.train_downstream import build_arg_parser as build_downstream_parser, run_benchmark

# Unified CLI entry point for running individual stages or full pipeline
def main():
    parser = argparse.ArgumentParser(description="Master pipeline runner for AQI project")
    parser.add_argument("--stage", choices=["impute", "evaluate", "train", "all"], default="all", help="Pipeline stage to execute")

    args, remaining_args = parser.parse_known_args()

    if args.stage in ["impute", "all"]:
        print("=== STAGE 1: IMPUTATION ===")
        impute_args, _ = build_impute_parser().parse_known_args(remaining_args)
        run_imputation(impute_args)

    if args.stage in ["evaluate", "all"]:
        print("\n=== STAGE 2: IMPUTATION EVALUATION ===")
        eval_args, _ = build_eval_parser().parse_known_args(remaining_args)
        run_evaluation(eval_args)

    if args.stage in ["train", "all"]:
        print("\n=== STAGE 3: DOWNSTREAM BENCHMARKING ===")
        downstream_args, _ = build_downstream_parser().parse_known_args(remaining_args)
        run_benchmark(downstream_args)

if __name__ == "__main__":
    main()

