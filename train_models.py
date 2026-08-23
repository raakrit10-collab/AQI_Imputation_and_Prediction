from src.pipeline.train_downstream import build_arg_parser, run_benchmark

# Main entry point for downstream model training and benchmarking script
def main():
    args = build_arg_parser().parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    main()
