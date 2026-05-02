from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import torch.utils.data as data

from ..utils import _log_api_usage_once

T_co = TypeVar("T_co", covariant=True)


class VisionDataset(data.Dataset[T_co], Generic[T_co]):
    """Base class for marrvision datasets with torchvision-style transforms."""

    _repr_indent = 4

    def __init__(
        self,
        root: Union[str, Path, None] = None,
        transforms: Optional[Callable[[Any, Any], tuple[Any, Any]]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        _log_api_usage_once(self)
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        has_joint_transform = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_joint_transform and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can be passed"
            )

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = (
            StandardTransform(transform, target_transform)
            if has_separate_transform
            else transforms
        )

    def __getitem__(self, index: int) -> T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if self.transforms is not None:
            body.append(repr(self.transforms))
        lines = [head] + [" " * self._repr_indent + line for line in body if line]
        return "\n".join(lines)

    def _format_transform_repr(
        self, transform: Callable[..., Any], head: str
    ) -> list[str]:
        lines = repr(transform).splitlines()
        return [f"{head}{lines[0]}"] + [
            f"{' ' * len(head)}{line}" for line in lines[1:]
        ]

    def extra_repr(self) -> str:
        return ""


class StandardTransform:
    def __init__(
        self,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(
        self, transform: Callable[..., Any], head: str
    ) -> list[str]:
        lines = repr(transform).splitlines()
        return [f"{head}{lines[0]}"] + [
            f"{' ' * len(head)}{line}" for line in lines[1:]
        ]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(
                self.target_transform, "Target transform: "
            )
        return "\n".join(body)
