"""
WM-811K Wafer Defect Classification Pipeline
Single entry point: python run_pipeline.py --config config.yaml
"""
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", default="all", help="all | eda | features | train | spc")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Config loaded: {args.config}")
    print(f"Stage: {args.stage}")
    # TODO: import and call stage functions from src/

if __name__ == "__main__":
    main()
