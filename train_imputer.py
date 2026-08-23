from src.pipeline.train_imputation import build_arg_parser, run_imputation

# Main entry point for imputer training script
def main():
    args = build_arg_parser().parse_args()
    run_imputation(args)

if __name__ == "__main__":
    main()
