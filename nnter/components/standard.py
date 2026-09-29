"""`Standard`, the envoy every component derives from, and the two helpers for tuple outputs."""

from __future__ import annotations

from typing import Any

import torch
from nnsight.intervention.envoy import Envoy

from .eproperty import EProperty


def first_tensor(value: Any) -> torch.Tensor:
    """The tensor of a module output: the first element when it is a tuple."""
    return value[0] if isinstance(value, tuple) else value


def rewrap(envoy: Envoy, value: torch.Tensor) -> Any:
    """``value`` in the shape the module returned: back in its tuple, if any."""
    current = envoy.output
    return (value, *current[1:]) if isinstance(current, tuple) else value


class Standard(Envoy):
    """An envoy carrying standard values: what `Layer`, `Attention` and `Mlp` share.

    A value of a child that reads an operation inside *this* module's forward
    (``"../source.<op>.output"``) needs the forward instrumented before the
    call begins: the read can come after the call started, when instrumenting
    is too late for it. So the forward is instrumented when the envoy is built,
    and again when real weights replace meta ones, which reinstalls the plain
    forward.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._instrument_for_children()

    def _update(self, module: Any) -> None:
        super()._update(module)
        self._instrument_for_children()

    def _instrument_for_children(self) -> None:
        for _, child in self._named_children():
            if isinstance(child, Standard) and any(
                value.reads_parent_forward() for value in type(child).values().values()
            ):
                self.source
                return

    @classmethod
    def values(cls) -> dict[str, EProperty]:
        """This class's standard values by name, base classes first."""
        found: dict[str, EProperty] = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if isinstance(attr, EProperty):
                    found[name] = attr
        return found

    def status(self) -> dict[str, str | None]:
        """Each standard value here -> ``None`` when available, else the reason."""
        return {name: value.reason(self) for name, value in self.values().items()}
