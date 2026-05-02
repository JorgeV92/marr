from __future__ import annotations

import gzip
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Union

import torch

from .vision import VisionDataset


def _read_file(path: Union[str, Path]) -> bytes:
    with open(path, "rb") as input_file:
        return input_file.read()


def _read_be_u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise RuntimeError("Unexpected end of IDX file")
    return int.from_bytes(data[offset : offset + 4], byteorder="big", signed=False)


def read_sn3_pascalvincent_tensor(
    path: Union[str, Path], strict: bool = True
) -> torch.Tensor:
    data = _read_file(path)
    if len(data) < 8:
        raise RuntimeError(f"IDX file is too small: {path}")

    magic = _read_be_u32(data, 0)
    data_type = (magic >> 8) & 0xFF
    dimensions = magic & 0xFF

    if dimensions < 1 or dimensions > 3:
        raise RuntimeError("Expected IDX tensor with 1 to 3 dimensions")
    if data_type != 8:
        raise RuntimeError("Only unsigned byte IDX tensors are supported")

    shape = [_read_be_u32(data, 4 * (index + 1)) for index in range(dimensions)]
    header_size = 4 * (dimensions + 1)
    value_count = 1
    for value in shape:
        value_count *= value

    expected_size = header_size + value_count
    if len(data) < expected_size or (strict and len(data) != expected_size):
        raise RuntimeError(
            f"IDX payload size mismatch for {path}: expected {expected_size} bytes, got {len(data)}"
        )

    payload = bytearray(data[header_size:expected_size])
    return torch.frombuffer(payload, dtype=torch.uint8).clone().reshape(shape)


def read_label_file(path: Union[str, Path]) -> torch.Tensor:
    tensor = read_sn3_pascalvincent_tensor(path, strict=False)
    if tensor.ndim != 1:
        raise RuntimeError("MNIST label file must contain a 1D tensor")
    return tensor.long()


def read_image_file(path: Union[str, Path]) -> torch.Tensor:
    tensor = read_sn3_pascalvincent_tensor(path, strict=False)
    if tensor.ndim != 3:
        raise RuntimeError("MNIST image file must contain a 3D tensor")
    return tensor


class MNIST(VisionDataset[tuple[torch.Tensor, int]]):
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    class_to_idx = {name: index for index, name in enumerate(classes)}

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[int], int]] = None,
        download: bool = False,
        transforms: Optional[
            Callable[[torch.Tensor, int], tuple[torch.Tensor, int]]
        ] = None,
        add_channel_dimension: bool = True,
    ) -> None:
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.train = train
        self.add_channel_dimension = add_channel_dimension

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. Place extracted MNIST files in {self.raw_folder}"
            )

        self.data, self.targets = self._load_data()
        if len(self.data) != len(self.targets):
            raise RuntimeError("MNIST image and label counts do not match")

    @property
    def raw_folder(self) -> Path:
        return Path(self.root) / "MNIST" / "raw"

    @property
    def processed_folder(self) -> Path:
        return Path(self.root) / "MNIST" / "processed"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.data[index]
        if self.add_channel_dimension:
            image = image.unsqueeze(0)
        target = int(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return int(self.data.size(0))

    def extra_repr(self) -> str:
        return f"Split: {'Train' if self.train else 'Test'}"

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = "train" if self.train else "t10k"
        data = read_image_file(self.raw_folder / f"{prefix}-images-idx3-ubyte")
        targets = read_label_file(self.raw_folder / f"{prefix}-labels-idx1-ubyte")
        return data, targets

    def _check_exists(self) -> bool:
        return all(
            (self.raw_folder / self._uncompressed_name(filename)).is_file()
            for filename, _ in self.resources
        )

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        for filename, md5 in self.resources:
            archive_path = self.raw_folder / filename
            for mirror in self.mirrors:
                try:
                    self._download_url(mirror + filename, archive_path)
                    if not self._check_md5(archive_path, md5):
                        archive_path.unlink(missing_ok=True)
                        raise RuntimeError(f"MD5 mismatch for {archive_path}")
                    self._extract_gzip(archive_path)
                    break
                except (OSError, RuntimeError, urllib.error.URLError):
                    archive_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"Could not download {filename} from any mirror")

    @staticmethod
    def _uncompressed_name(filename: str) -> str:
        return filename[:-3] if filename.endswith(".gz") else filename

    @staticmethod
    def _download_url(url: str, destination: Path) -> None:
        with urllib.request.urlopen(url) as response, open(
            destination, "wb"
        ) as output_file:
            output_file.write(response.read())

    @staticmethod
    def _check_md5(path: Path, expected: str) -> bool:
        digest = hashlib.md5()
        with open(path, "rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected

    @staticmethod
    def _extract_gzip(path: Path) -> None:
        output_path = path.with_suffix("")
        with gzip.open(path, "rb") as input_file, open(
            output_path, "wb"
        ) as output_file:
            output_file.write(input_file.read())
