"""The base envoys and the descriptor every family builds its toolkit from.

A family gives its modules standard *values* by wrapping them in `Envoy`
subclasses (nnsight's ``envoys=``): `Layer` on the decoder block, `Attention`
on the softmax-attention module, `Mlp` on the feed-forward, and
`LinearAttention` on a hybrid's gated DeltaNet mixer and `StateSpace` on a
Mamba-2 (SSD) mixer, both `RecurrentMixer`s (the mechanism every recurrent
mixer shares). The families under
`nnter.families` subclass these, so a family whose forward is spelled
differently overrides only what differs, and key them on its own module types
in its ``ENVOYS``.

A value a checkpoint does not have says so: every descriptor here takes
``unavailable=`` (a reason, or a function of the envoy returning one), reading
it raises `Unavailable` with the reason before the model runs, and
`Standard.status` / `StandardizedTransformer.status` report the reasons
without raising. A family without a value assigns `unavailable("...")`.

The three boundary values mean the same thing on every family:
``layer_output`` is the residual stream leaving the block, and
``attention_output`` / ``mlp_output`` are the *contributions* the sublayers
add to it, so ``layer.input + attention_output + mlp_output == layer_output``
whether the block is sequential or parallel. On a family whose sublayer adds
the residual inside the module (BLOOM, MPT, DBRX) or adds a post-sublayer
norm's output instead (Gemma-2/3, OLMo-2), the family's subclass points the
value at the right place; the identity is what the tests check.

One descriptor, `EProperty`, whose key is a path from the host envoy:

* ``"output"``: the host's own output, a view over the same location as
  ``.output``, so reading, writing and in-place edits all go through the
  interleaver the way ``.output`` does (`Layer.layer_output`).
* ``"../post_attention_layernorm.output"``, ``"embed_tokens.output"``: a value
  produced by a module named relative to this one (a sandwich block's
  post-sublayer norm, the root's embedding).
* ``"source.attention_interface_1.inputs"``: an operation inside a forward,
  reached through ``.source``, optionally one element of it (``select``).
* Computed from several served values, a `DerivedEProperty` (a DeltaNet
  layer's state after every token). An operation inside a called
  function only exists once someone has drilled into that call **in the
  current run** — the interleaver resolves the callee from the live value and
  clears what it built at the start of every run — so this descriptor drills
  in before every read or write, the step a plain eproperty has no place for
  (`Attention.attention_probabilities`).
"""

from .attention import (
    INTERFACE, NOT_ON_INTERFACE, Attention, HeadOutputs, Keys, Pattern, Queries, Values, interface_reason, needs_eager,
    seq_first,
)
from .eproperty import (
    DerivedEProperty, EProperty, Unavailable, branched,
    per_call, unavailable,
)
from .layer import Layer, Residual
from .linear_attention import Gates, LinearAttention, LinearQK, LinearV
from .mlp import Mlp
from .recurrent import (
    RecurrentMixer, State, States, at_occurrence, needs_recurrent_routing, needs_torch_kernels, route_delta_rule,
    route_kernels,
)
from .standard import Standard, first_tensor, rewrap
from .state_space import SSDHeadOutputs, SSDKeys, SSDQueries, SSDValues, StateSpace

__all__ = [
    "Attention", "DerivedEProperty", "EProperty", "Gates", "HeadOutputs", "INTERFACE", "Keys", "Layer", "LinearAttention",
    "LinearQK", "LinearV", "Mlp", "Pattern", "Queries", "RecurrentMixer", "Residual", "State", "States", "Values",
    "NOT_ON_INTERFACE", "SSDHeadOutputs", "SSDKeys", "SSDQueries", "SSDValues", "Standard", "StateSpace", "Unavailable",
    "at_occurrence",
    "branched", "first_tensor", "interface_reason", "needs_eager", "needs_recurrent_routing",
    "needs_torch_kernels", "per_call", "rewrap", "route_delta_rule", "route_kernels", "seq_first", "unavailable",
]
