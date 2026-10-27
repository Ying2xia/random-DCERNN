"""Small runtime-compatibility helpers."""

from __future__ import annotations

from itertools import zip_longest
from typing import Any, Iterator


def zip_strict(*iterables: Any) -> Iterator[tuple[Any, ...]]:
    """Yield equal-length iterables and raise if their lengths differ.

    Equivalent to ``zip(*iterables, strict=True)`` but compatible with
    Python 3.9, whose built-in ``zip`` accepts no keyword arguments.
    """

    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments must have equal length")
        yield values
