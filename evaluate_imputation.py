from src.pipeline.eval_imputation import build_arg_parser, run_evaluation

# Main entry point for imputation evaluation script
def main():
    args = build_arg_parser().parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
