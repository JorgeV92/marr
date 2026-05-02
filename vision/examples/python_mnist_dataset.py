from __future__ import annotations

import argparse

from marrvision.datasets import MNIST


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Dataset root containing MNIST/raw")
    parser.add_argument(
        "--download", action="store_true", help="Download and extract MNIST"
    )
    args = parser.parse_args()

    dataset = MNIST(args.root, train=True, download=args.download)
    image, target = dataset[0]

    print(f"MNIST train samples: {len(dataset)}")
    print(f"First image shape: {tuple(image.shape)}")
    print(f"First target: {target}")


if __name__ == "__main__":
    main()
