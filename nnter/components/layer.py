"""`Layer`: a decoder block, whose residual stream is ``layer_output``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from nnsight.intervention.envoy import Envoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import EProperty
from .standard import Standard, first_tensor, rewrap

if TYPE_CHECKING:
    from .attention import Attention
    from .linear_attention import LinearAttention
    from .mlp import Mlp

#: The residual stream and everything added to it: ``layer_output``, the contributions, ``token_embeddings``, a sublayer's input.
Residual = Float[Tensor, "batch seq hidden"]


class Layer(Standard):
    """A decoder block. Its hidden states are ``layer_output``, whatever the block returns.

    Attributes:
        self_attn: The softmax attention, an `Attention`; absent on a hybrid's linear blocks.
        linear_attn: The recurrent mixer, a `LinearAttention` (gated DeltaNet) or a `StateSpace` (Mamba-2);
            hybrids and state-space models only.
        mlp: The feed-forward, an `Mlp`; absent on OPT.
        input_layernorm, post_attention_layernorm: The block's norms under their aliased names,
            where the family has them (their meaning varies; see the README).
    """

    self_attn: Attention
    linear_attn: LinearAttention
    mlp: Mlp
    input_layernorm: Envoy
    post_attention_layernorm: Envoy

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bind_child_aliases()

    def _update(self, module: Any) -> None:
        super()._update(module)
        self._bind_child_aliases()

    def child_aliases(self) -> dict[str, str]:
        """Standard names this block binds on its own children, decided per block: native child name -> alias.

        ``RENAME`` binds a name wherever it resolves, the same on every block.
        A family whose blocks hold one child under one native name whatever
        it is (Nemotron-H's ``mixer``: a Mamba-2 mixer, attention, a mixture of
        experts or an MLP) answers here from the child itself, so the standard
        name follows what the block holds; a family whose native name would
        also match inside the children (Mamba-2's ``norm``, which the mixer has
        too) binds it here for the block alone. Bound like a ``RENAME`` alias
        (an attribute on the block, recorded in ``_aliases``), at build and
        again when real weights replace meta ones.
        """
        return {}

    def _bind_child_aliases(self) -> None:
        for alias in self.__dict__.pop("_child_aliases", ()):  # a rebind after `_update` drops the previous ones first
            self.__dict__.pop(alias, None)
            self._aliases.pop(alias, None)
        bound = []
        for name, alias in self.child_aliases().items():
            target = self._child_map.get(name)
            if target is None:
                continue
            if alias in self.__dict__ or alias in self._aliases:
                raise ValueError(f"child alias {alias!r} for {name!r} would shadow a name already on `{self.path}`")
            object.__setattr__(self, alias, target)
            self._aliases[alias] = name
            bound.append(alias)
        self.__dict__["_child_aliases"] = bound

    #: Whether the block returns ``(hidden_states, ...)`` rather than the tensor alone.
    #: What `skip_with` has to hand back in the block's place; a family that
    #: returns a tuple says so.
    returns_tuple = False

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` on as its residual stream.

        The block does not run; ``hidden`` takes the place of its
        ``layer_output``, packed the way the block would have returned it (a
        tuple family gets ``(hidden, None)``: the second element is the
        attention weights nothing downstream reads). Call it inside a trace,
        before the block runs.
        """
        self.skip((hidden, None) if self.returns_tuple else hidden)

    @EProperty(key="output", description="The residual stream leaving the block, a tensor even when the block returns a tuple")
    def layer_output(self, value: Any) -> Residual:
        """The residual stream leaving this block, always a tensor.

        Some blocks return ``hidden_states`` alone, others a tuple with it first
        (GPT-J, Bloom, MPT, Falcon). This is the tensor either way, the same
        object the block returned, so in-place edits reach the model. Assigning
        replaces it and, for a tuple block, leaves the other elements as they
        were::

            with model.trace(prompt):
                resid = model.layers[3].layer_output.save()
                model.layers[3].layer_output[:, -1] = 0
                model.layers[4].layer_output = resid * 2
        """
        return first_tensor(value)

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)
