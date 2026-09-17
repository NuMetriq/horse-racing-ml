import argparse

import kagglehub


DATASET = "deltaromeo/horse-racing-results-ukireland-2015-2025"


def main():
    parser = argparse.ArgumentParser(
        description="Download the horse-racing dataset from Kaggle."
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Specific dataset version; omit to request the latest",
    )
    args = parser.parse_args()

    if args.version is not None and args.version < 1:
        parser.error("--version must be a positive integer")

    if args.version is None:
        handle = DATASET
        print("Requested version: latest")
    else:
        handle = f"{DATASET}/versions/{args.version}"
        print(f"Requested version: {args.version}")

    download_path = kagglehub.dataset_download(handle)

    print(f"Dataset available at: {download_path}")


if __name__ == "__main__":
    main()