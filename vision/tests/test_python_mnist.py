from __future__ import annotations

import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marrvision.datasets import MNIST  # noqa: E402
from marrvision.datasets.vision import StandardTransform, VisionDataset  # noqa: E402


def _write_idx(
    path: Path, magic: int, dimensions: tuple[int, ...], values: bytes
) -> None:
    with open(path, "wb") as output:
        output.write(struct.pack(">I", magic))
        for value in dimensions:
            output.write(struct.pack(">I", value))
        output.write(values)


class PythonMNISTTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="marrvision_python_mnist_"))
        raw_folder = self.root / "MNIST" / "raw"
        raw_folder.mkdir(parents=True)

        _write_idx(
            raw_folder / "train-images-idx3-ubyte",
            0x00000803,
            (2, 2, 3),
            bytes(range(12)),
        )
        _write_idx(
            raw_folder / "train-labels-idx1-ubyte", 0x00000801, (2,), bytes([7, 3])
        )
        _write_idx(
            raw_folder / "t10k-images-idx3-ubyte",
            0x00000803,
            (2, 2, 3),
            bytes(range(12, 24)),
        )
        _write_idx(
            raw_folder / "t10k-labels-idx1-ubyte", 0x00000801, (2,), bytes([2, 1])
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_reads_tensor_first_mnist(self) -> None:
        dataset = MNIST(self.root, train=True)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.data.shape, torch.Size([2, 2, 3]))
        self.assertEqual(dataset.targets.tolist(), [7, 3])
        self.assertEqual(dataset.class_to_idx["7 - seven"], 7)

        image, target = dataset[1]
        self.assertEqual(image.shape, torch.Size([1, 2, 3]))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertEqual(int(image[0, 0, 0]), 6)
        self.assertEqual(target, 3)

    def test_applies_separate_transforms(self) -> None:
        dataset = MNIST(
            self.root,
            transform=lambda image: image.float().div(255.0),
            target_transform=lambda target: target + 10,
        )

        image, target = dataset[0]
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(target, 17)

    def test_applies_joint_transform(self) -> None:
        def transform(image: torch.Tensor, target: int) -> tuple[torch.Tensor, int]:
            return image.add(1), target - 1

        dataset = MNIST(self.root, transforms=transform)
        image, target = dataset[0]

        self.assertEqual(int(image[0, 0, 0]), 1)
        self.assertEqual(target, 6)

    def test_base_dataset_rejects_mixed_transform_styles(self) -> None:
        with self.assertRaises(ValueError):
            VisionDataset(
                self.root, transforms=lambda x, y: (x, y), transform=lambda x: x
            )

    def test_standard_transform_repr(self) -> None:
        transform = StandardTransform(
            transform=lambda x: x, target_transform=lambda y: y
        )

        self.assertIn("Transform:", repr(transform))
        self.assertIn("Target transform:", repr(transform))


if __name__ == "__main__":
    unittest.main()
