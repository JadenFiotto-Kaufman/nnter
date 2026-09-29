"""`Mlp`: a feed-forward module, whose contribution is ``mlp_output``."""

from __future__ import annotations

from typing import Any

import torch

from .eproperty import EProperty
from .layer import Residual
from .standard import Standard, first_tensor, rewrap


class Mlp(Standard):
    """A feed-forward module. Its contribution is ``mlp_output``."""

    @EProperty(key="output", description="What the MLP adds to the residual stream")
    def mlp_output(self, value: Any) -> Residual:
        """The MLP sublayer's contribution to the residual stream.

        A tensor even when the module returns a tuple (a mixture of experts
        returns router scores beside it). On a family whose MLP adds the
        residual inside the module, the family's subclass reads the
        pre-residual value instead. In-place edits and assignment reach the
        model.
        """
        return first_tensor(value)

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)
