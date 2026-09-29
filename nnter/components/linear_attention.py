"""`LinearAttention`: a gated DeltaNet mixer, its recurrent state, and the kernel it runs through."""

from __future__ import annotations

from typing import Any, Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import SourceEnvoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty, SourceEProperty, Unavailable, at_occurrence, branched, per_call
from .layer import Residual
from .standard import Standard, first_tensor, rewrap

#: The layouts at the delta-rule call, tokens before heads: the queries and keys on the state's key side, the
#: values and each head's read of the state on its value side, one gate per token and head, and the state itself,
#: one ``key_dim`` by ``value_dim`` matrix per head, alone or stacked along the tokens.
LinearQK = Float[Tensor, "batch seq heads key_dim"]
LinearV = Float[Tensor, "batch seq heads value_dim"]
Gates = Float[Tensor, "batch seq heads"]
State = Float[Tensor, "batch heads key_dim value_dim"]
States = Float[Tensor, "batch seq heads key_dim value_dim"]


def _modeling_module(envoy: Envoy):
    import sys

    return sys.modules[type(envoy._module).__module__]


def _delta_rule_loop(module) -> Callable:
    """transformers' pure-torch, token-by-token gated delta rule behind the family's kernel names.

    Each name is bound to a dispatcher that picks an optimized kernel when one
    is installed and the torch loop otherwise; the loop itself sits in the
    dispatcher's closure as ``torch_function``. A name already bound to the
    loop (after `route_delta_rule`) is returned as is.
    """
    import inspect

    bound = getattr(module, LinearAttention.RECURRENT_KERNEL.removesuffix("_0"))
    closure = inspect.getclosurevars(bound).nonlocals if getattr(bound, "__closure__", None) else {}
    return closure.get("torch_function", bound)


def _mixer_module(family):
    """The transformers modeling module a hybrid family's DeltaNet mixer lives in."""
    import sys
    from types import ModuleType

    if isinstance(family, ModuleType) and not hasattr(family, "ENVOYS"):
        return family  # already the modeling module
    mixer = next((cls for cls, envoy in family.ENVOYS.items() if issubclass(envoy, LinearAttention)), None)
    if mixer is None:
        raise ValueError(f"{family.__name__} has no gated DeltaNet mixer to route")
    return sys.modules[mixer.__module__]


def route_delta_rule(family, kernel: str = "recurrent") -> None:
    """Bind a hybrid family's delta-rule kernels, process-wide.

    ``"recurrent"`` binds both the prompt's and the decode step's names in
    the family's modeling module to transformers' token-by-token torch loop:
    the slower kernel, and the only one that materializes the state after
    every token (`LinearAttention.state`, `states`, `state_after`,
    `set_state_after`). ``"chunked"`` restores what the module bound at
    import. Like installing a kernel, it applies to every model of the
    family; call it before tracing a layer, since a forward ``.source`` has
    already instrumented keeps the binding it was compiled with. ``family``
    is the family module (``model.family``, or ``nnter.families.qwen3_5_text``) or its modeling module.
    """
    module = _mixer_module(family)
    originals = module.__dict__.setdefault("_nnter_delta_rules", {
        name: getattr(module, name) for name in (LinearAttention.CHUNK_KERNEL.removesuffix("_0"), LinearAttention.RECURRENT_KERNEL.removesuffix("_0"))
    })
    if kernel == "recurrent":
        loop = _delta_rule_loop(module)
        for name in originals:
            setattr(module, name, loop)
    elif kernel == "chunked":
        for name, original in originals.items():
            setattr(module, name, original)
    else:
        raise ValueError(f"delta_rule must be 'recurrent' or 'chunked', not {kernel!r}")


def needs_torch_kernels(envoy: Envoy) -> str | None:
    """Why the DeltaNet values are unavailable: the kernel has no Python source to read inside."""
    import inspect

    module = _modeling_module(envoy)
    for name in (LinearAttention.CHUNK_KERNEL, LinearAttention.RECURRENT_KERNEL):
        bound = getattr(module, name.removesuffix("_0"), None)
        closure = inspect.getclosurevars(bound).nonlocals if getattr(bound, "__closure__", None) else {}
        if "implementation" in closure and closure["implementation"] is not closure["torch_function"]:
            return (
                "read inside transformers' pure-torch gated delta rule, but this process dispatches "
                "to an optimized kernel (flash-linear-attention / causal-conv1d) with no Python "
                "source; uninstall it to read these"
            )
    return None


def needs_recurrent_routing(envoy: Envoy) -> str | None:
    """Why the per-token state is unavailable: only the token-by-token kernel materializes it."""
    reason = needs_torch_kernels(envoy)
    if reason:
        return reason
    module = _modeling_module(envoy)
    loop = _delta_rule_loop(module)
    if getattr(module, LinearAttention.CHUNK_KERNEL.removesuffix("_0")) is not loop:
        return (
            "the state after each token is materialized only by the token-by-token kernel; "
            "the chunked kernel a prompt runs through keeps one state per 64 tokens. Call "
            "nnter.route_delta_rule(model.family, 'recurrent') before tracing this layer "
            "(slower, like attn_implementation='eager')"
        )
    return None

class LinearAttention(Standard):
    """A gated DeltaNet mixer (Qwen3-Next, Qwen3.5/3.6): linear attention with a recurrent state.

    It projects queries, keys and values like attention, but mixes them
    through a per-head recurrent state instead of a softmax over keys: for
    each token the state decays by a learned gate (``decays``), takes up the
    new key/value pair scaled by ``betas``, and the query reads against it.
    So the shared names mean what they mean on `Attention` — the contribution,
    the queries/keys/values the mixer receives, the per-head outputs — and
    there is no pattern and no scores; instead there are the gate, the beta,
    and the state entering and leaving the layer.

    Everything but ``attention_output`` is read at the delta-rule kernel call
    (`KERNEL`). A prompt runs the chunked rule and each decode step of
    ``generate`` the recurrent rule; the forward binds ``use_precomputed_states``
    before it branches, and `KERNEL` reads that to name the call that fires
    on this step, so the same values work in a ``trace`` and at every step of
    ``tracer.iter``. On a decode step the sequence axis is 1 and
    ``state_input`` is the cached state. The kernels have to be transformers'
    pure-torch ones (see `needs_torch_kernels`).

    The state *after every token* of a prompt exists only in the
    token-by-token kernel: the chunked one a prompt runs through carries the
    state between 64-token chunks. Like eager attention, that is the user's
    choice: ``nnter.route_delta_rule(model.family, "recurrent")``
    routes the family's prompts through the token-by-token loop, and then
    `state`, `states`, `state_after` and `set_state_after` read and write the
    state at any position; without it, reading one raises `Unavailable` with
    that instruction and `status` reports it.
    """

    #: The call a prompt runs through: ``torch_chunk_gated_delta_rule(query, key, value, g=, beta=, initial_state=, ...)``.
    CHUNK_KERNEL = "torch_chunk_gated_delta_rule_0"
    #: The call each decode step runs through, with the same arguments.
    RECURRENT_KERNEL = "torch_recurrent_gated_delta_rule_0"
    #: Whichever of the two fires on this call, chosen by the forward's own branch variable.
    KERNEL = branched("use_precomputed_states_0", {False: CHUNK_KERNEL, True: RECURRENT_KERNEL})
    #: Inside the token-by-token kernel, the binding of the state after each token's update.
    STATE_OP = "last_recurrent_state_3"

    @EProperty(key="output", description="What the linear attention adds to the residual stream")
    def attention_output(self, value: Any) -> Residual:
        """The mixer's contribution to the residual stream, a tensor."""
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)

    @SourceEProperty(KERNEL, attribute="inputs", select=0, description="The queries entering the delta rule, [batch, seq, heads, key_dim]", unavailable=needs_torch_kernels)
    def attention_queries(self, value: torch.Tensor) -> LinearQK:
        """The queries the delta rule receives, ``[batch, seq, heads, key_dim]``: after the conv, the activation and the repeat to ``num_v_heads``."""
        return value

    @SourceEProperty(KERNEL, attribute="inputs", select=1, description="The keys entering the delta rule, [batch, seq, heads, key_dim]", unavailable=needs_torch_kernels)
    def attention_keys(self, value: torch.Tensor) -> LinearQK:
        """The keys the delta rule receives, ``[batch, seq, heads, key_dim]``."""
        return value

    @SourceEProperty(KERNEL, attribute="inputs", select=2, description="The values entering the delta rule, [batch, seq, heads, value_dim]", unavailable=needs_torch_kernels)
    def attention_values(self, value: torch.Tensor) -> LinearV:
        """The values the delta rule receives, ``[batch, seq, heads, value_dim]``."""
        return value

    @SourceEProperty(KERNEL, attribute="inputs", select="g", description="The per-token log decay of the recurrent state, [batch, seq, heads]", unavailable=needs_torch_kernels)
    def decays(self, value: torch.Tensor) -> Gates:
        """The gate: the log of how much of the state each token keeps, ``[batch, seq, heads]``, float32 and non-positive."""
        return value

    @SourceEProperty(KERNEL, attribute="inputs", select="beta", description="The per-token write strength into the state, [batch, seq, heads]", unavailable=needs_torch_kernels)
    def betas(self, value: torch.Tensor) -> Gates:
        """How strongly each token's key/value pair is written into the state, ``[batch, seq, heads]``, in ``(0, 1)`` (``(0, 2)`` on OLMo-Hybrid with ``linear_allow_neg_eigval``)."""
        return value

    @SourceEProperty(KERNEL, attribute="inputs", select="initial_state", description="The recurrent state entering the layer, [batch, heads, key_dim, value_dim], or None at the start of a prompt (a copy of the cache's buffer)", unavailable=needs_torch_kernels)
    def state_input(self, value: Any) -> State | None:
        """The state this call starts from: ``None`` on a fresh prompt, the cached state on a decode step.

        A copy: the cache hands the kernel its own buffer and overwrites it in
        place with the step's new state afterwards, so the live tensor would
        read as this step's *output* by the time the trace ends. Assign to
        replace what the step starts from.
        """
        return value if value is None else value.clone()

    @SourceEProperty(KERNEL, attribute="output", select=0, description="The per-head outputs before the gated norm and the output projection, [batch, seq, heads, value_dim]", unavailable=needs_torch_kernels)
    def attention_head_outputs(self, value: torch.Tensor) -> LinearV:
        """Each head's read of the state, ``[batch, seq, heads, value_dim]``, before the gated norm and ``out_proj``."""
        return value

    @SourceEProperty(KERNEL, attribute="output", select=1, description="The recurrent state leaving the layer, [batch, heads, key_dim, value_dim]", unavailable=needs_torch_kernels)
    def state_output(self, value: torch.Tensor) -> State:
        """The state after this call's last token, ``[batch, heads, key_dim, value_dim]``: what the next decode step starts from."""
        return value

    # -- the state at every token: the token-by-token kernel only -------------------

    @staticmethod
    def _token_state_op(envoy: Envoy) -> str:
        # Unpinned, or pinned to index 0, a read decides the kernel like every
        # other value (once per call, cached), so a call-level value read after
        # a token loop still finds the branch decided. Pinned to a later token
        # index it cannot read the branch variable — that fires once per call —
        # so it takes the prompt's kernel, the one a token loop walks.
        from nnsight.intervention.interleaver import Mediator

        step = Mediator.current("state").iteration
        kernel = type(envoy).KERNEL(envoy) if step in (None, 0) else LinearAttention.CHUNK_KERNEL
        return f"{kernel}.source.{LinearAttention.STATE_OP}"

    @SourceEProperty(_token_state_op, description="The recurrent state after one token of the prompt; iterate it with tracer.iter; needs route_delta_rule(family, 'recurrent')", unavailable=needs_recurrent_routing)
    def state(self, value: torch.Tensor) -> State:
        """The state after a token of the prompt, ``[batch, heads, key_dim, value_dim]``: one occurrence per token.

        A location inside the token-by-token kernel, so it takes nnsight's
        own iteration: ``for t in tracer.iter[:]: mix.state`` reads the state
        after every token, ``tracer.iter[4]`` the one after token 4, and an
        assignment there is a write the following tokens continue from.
        Outside any ``tracer.iter`` a read is the state after token 0. It
        lives on the prompt's kernel call: under ``generate`` walk it in an
        inner ``tracer.iter`` on step 0, and take a decode step's state from
        `state_output`, one token per step. Needs ``route_delta_rule(family, "recurrent")``.
        """
        return value

    def _call(self) -> tuple[int, int]:
        """This call's sequence length and the occurrence its first token is, decided once per call.

        The kernel's queries are served once per call: reading them parks the
        worker at the call's start, which is the one moment the state op's
        occurrence count is the number of tokens *earlier* calls put through
        it. Occurrences are counted per location over the whole run, so a
        decode step's single token is not occurrence 0 once an earlier step
        has fired the recurrent kernel's op. Whichever per-token read comes
        first in the call takes both numbers; the rest reuse them.
        """
        from nnsight.intervention.interleaver import Mediator

        def compute():
            seq = self.attention_queries.shape[1]
            op = getattr(getattr(self.source, type(self).KERNEL(self)).source, self.STATE_OP)
            location = f"{op.path}.output"
            return seq, Mediator.current(location).occurrence(location)

        return per_call(self, "call", compute)

    def _seq(self) -> int:
        return self._call()[0]

    def _token_op(self) -> tuple[SourceEnvoy, int]:
        """This call's state-update op and the occurrence its first token is (see `_call`)."""
        _, first = self._call()
        op = getattr(getattr(self.source, type(self).KERNEL(self)).source, self.STATE_OP)
        return op, first

    def _states(self) -> States:
        # Through whichever kernel call fires on this step, so a decode step's
        # one-token call answers too; `state` alone is the prompt's location.
        seq = self._seq()
        op, first = self._token_op()
        states = []
        for t in range(seq):
            for _ in at_occurrence(first + t):
                states.append(op.output)
        return torch.stack(states, dim=1)

    #: The state after each token, ``[batch, seq, heads, key_dim, value_dim]``: one occurrence of
    #: the token-by-token kernel's state update per token, so it needs ``route_delta_rule(family, "recurrent")``.
    #: The last is `state_output`. Read-only, a stack of copies: `set_state_after` writes one position.
    states = DerivedEProperty(
        _states,
        description="The recurrent state after every token of this call, [batch, seq, heads, key_dim, value_dim]; needs route_delta_rule(family, 'recurrent')",
        unavailable=needs_recurrent_routing,
    )

    def _require_state(self, name: str) -> None:
        reason = type(self).state.reason(self)
        if reason:
            raise Unavailable(f"{self.path}.{name} is not available: {reason}")

    def state_after(self, t: int) -> torch.Tensor:
        """The state after token ``t`` of this call: ``for _ in tracer.iter[t]: mix.state`` on a prompt, as a call that also counts from the call's own first token on a decode step."""
        self._require_state("state_after")
        op, first = self._token_op()
        for _ in at_occurrence(first + t):
            return op.output
        raise RuntimeError(f"this call had no token {t}")

    def set_state_after(self, t: int, value: torch.Tensor) -> None:
        """Assign `state` at token ``t``: the tokens after it continue from ``value``.

        A write inside the prompt. Reads follow the forward: in the same
        trace, read positions before ``t`` before the write and positions
        after it afterwards; `states` (every position) only before it.
        """
        self._require_state("set_state_after")
        op, first = self._token_op()
        for _ in at_occurrence(first + t):
            op.output = value
