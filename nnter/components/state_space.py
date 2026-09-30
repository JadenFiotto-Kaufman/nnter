"""`StateSpace`: a Mamba-2 (SSD) mixer's values, at the scan kernel call."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty, Unavailable, per_call
from .linear_attention import Gates
from .recurrent import RecurrentMixer, State, kernel, needs_torch_kernels

#: The layouts at the scan call, tokens before heads. SSD's ``C`` and ``B`` are projected once per *group* of
#: heads (``n_groups``, which divides ``num_heads``) and live on the state's ``state_dim`` side; the values
#: ``x`` and each head's read of the state are per head, ``head_dim`` wide. The state itself is the base's
#: `State` (``batch heads key_dim value_dim``): here ``key_dim`` is ``state_dim`` and ``value_dim`` is
#: ``head_dim``.
SSDQueries = Float[Tensor, "batch seq groups state_dim"]
SSDKeys = Float[Tensor, "batch seq groups state_dim"]
SSDValues = Float[Tensor, "batch seq heads head_dim"]
SSDHeadOutputs = Float[Tensor, "batch seq heads head_dim"]


def _this_call(envoy: Any) -> dict[str, Any]:
    """What is known about this call, decided at its first read and kept for the call's other reads.

    The kernel that fires, and once read, the call's arguments. `per_call`
    tells a new step by the step's pinned first read; a step whose first read
    is something else (the mixer's own ``.input``) relaxes the pin before a
    kernel value is read, so the record also carries how many of the mixer's
    calls had returned when it was made, and a record from an earlier call is
    made again.
    """
    from nnsight.intervention.interleaver import Mediator

    cls = type(envoy)
    location = f"{envoy.path}.output"  # passed once per call, after every value this record serves
    record = per_call(envoy, "kernel", dict)
    mediator = Mediator.current(location)
    if "kernel" not in record or record["calls"] != mediator.occurrence(location):
        record.clear()
        cached = getattr(envoy.source, cls.BRANCH).output
        # The length only matters over a cached state; read inside the forward, after the binding, so a
        # read of the mixer's own `.input` earlier in the step is not passed.
        one = cached and getattr(envoy.source, cls.SEQ_OP).output.shape[1] == 1
        record["kernel"] = cls.RECURRENT_KERNEL if one else cls.CHUNK_KERNEL
        record["calls"] = mediator.occurrence(location)
    return record


def _kernel(envoy: Any) -> str:
    """The scan call that fires on this call: the token-by-token update only for one token over a cached state.

    The forward decodes through ``mamba2_selective_state_update`` when
    ``use_precomputed_states and seq_len == 1`` and runs ``mamba2_chunk_scan``
    otherwise (a prompt, or several tokens over a cached state), so the choice
    reads the binding and, over a cached state, the call's sequence length
    (`SEQ_OP`), once per call.
    """
    return _this_call(envoy)["kernel"]


def argument(name: str):
    """A `select` function: where the scan call that fires takes ``name`` (the two kernels' call sites differ)."""

    def select(envoy: Any) -> int | str:
        return envoy._arguments_table()[name]

    select.__name__ = f"argument({name})"
    return select


def _state_output(envoy: Any) -> str:
    """The state the call leaves: the chunk scan returns it; the update writes it into the cache's buffer, so it is read inside."""
    cls = type(envoy)
    op = cls.KERNEL(envoy)
    if op == cls.RECURRENT_KERNEL:
        return f"source.{op}.source.{cls.UPDATED_STATE}.output"
    if not envoy._returns_state():
        raise Unavailable(
            f"{envoy.path}.state_output is not available: this call ran without a cache (use_cache=False), "
            "so the chunk scan returned no final state"
        )
    return f"source.{op}.output"


def _state_output_select(envoy: Any) -> int | None:
    return None if type(envoy).KERNEL(envoy) == type(envoy).RECURRENT_KERNEL else 1


def _head_outputs_select(envoy: Any) -> int | None:
    """The scan output alone: the chunk scan returns ``(output, final_state)`` when the call has a cache."""
    cls = type(envoy)
    return 0 if cls.KERNEL(envoy) == cls.CHUNK_KERNEL and envoy._returns_state() else None


class StateSpace(RecurrentMixer):
    """A Mamba-2 (SSD) mixer (Mamba-2, Nemotron-H, Bamba, Falcon-H1, Zamba2, GraniteMoeHybrid): a selective state space.

    SSD, the state-space dual, is linear attention with a scalar decay per
    head. Per head, with the state ``h`` a ``head_dim`` by ``state_dim``
    matrix, each token does::

        h = exp(dt * A) * h + dt * x B^T
        y = h C + D * x

    so the shared names mean what they mean on a gated DeltaNet
    (`LinearAttention`):

    * ``attention_queries`` is ``C``, what reads the state;
      ``attention_keys`` is ``B``, where a token writes into it; both
      ``[batch, seq, groups, state_dim]``, one per group of heads
      (``n_groups``), shared by the heads in the group.
    * ``attention_values`` is ``x`` (``hidden_states`` at the call), what is
      written, ``[batch, seq, heads, head_dim]``.
    * ``betas`` is ``dt = softplus(dt + dt_bias)``, the write strength,
      ``[batch, seq, heads]``; ``decays`` is ``A * dt``, the log of how much of
      the state each token keeps, ``[batch, seq, heads]``, float32 and
      non-positive. Both derived from the call's ``dt``, ``dt_bias`` and ``A``,
      so read-only: assign ``dt`` through the kernel's own arguments.
    * ``state_input`` / ``state_output``: the state entering and leaving the
      call, served as ``[batch, heads, state_dim, head_dim]`` (key side
      first, the shared `State` layout), the transpose of the cache's
      ``[batch, heads, head_dim, state_dim]``; an assignment is transposed
      back. ``state_input`` is ``None`` on a fresh prompt.
    * ``attention_head_outputs`` is ``y``, the scan's output with the ``D``
      skip, before the gated norm and ``out_proj``,
      ``[batch, seq, heads, head_dim]``.

    Everything but ``attention_output`` is read at the scan call:
    transformers' ``mamba2_chunk_scan`` on a prompt, its
    ``mamba2_selective_state_update`` on a decode step (`_kernel`). The two
    take their arguments in different places and the update has no sequence
    axis, so each value selects by the kernel that fires (`argument`) and a
    decode step's tensors are served with a sequence axis of 1, removed again
    on assignment. The update writes the new state into the cache's buffer
    and returns only ``y``, so a decode step's ``state_output`` is read inside
    it, at ``UPDATED_STATE``.

    The chunk scan carries the state per chunk and the update runs one
    token, so the state after every token of a prompt is not materialized:
    ``state``, ``states``, ``state_after`` and ``set_state_after`` are
    unavailable (``STATE_OP`` is ``None``). With ``mamba_ssm`` installed the
    kernels have no Python source; ``route_kernels(model.family, "torch")``
    binds transformers' pure-torch ones (`RecurrentMixer`).
    """

    #: The call a prompt runs through: ``mamba2_chunk_scan(hidden_states, dt, A, B, C, chunk_size=, D=, dt_bias=, initial_states=, ...)``.
    CHUNK_KERNEL = "mamba2_chunk_scan_0"
    #: The call each decode step runs through: ``mamba2_selective_state_update(state, hidden_states, dt, A, B, C, D, dt_bias=, ...)``.
    RECURRENT_KERNEL = "mamba2_selective_state_update_0"
    #: Neither kernel binds the state once per token.
    STATE_OP = None
    #: The first op after `BRANCH` whose output is ``[batch, seq, ...]`` on both paths: the input with padding masked.
    SEQ_OP = "apply_mask_to_padding_states_0"
    #: Inside the update, the binding of the new state before it is copied into the cache.
    UPDATED_STATE = "ssm_states_0"
    #: Where each kernel's call site passes each argument: a position, or a keyword.
    CHUNK_ARGUMENTS = {"hidden_states": 0, "dt": 1, "A": 2, "B": 3, "C": 4, "dt_bias": "dt_bias", "state": "initial_states"}
    RECURRENT_ARGUMENTS = {"state": 0, "hidden_states": 1, "dt": 2, "A": 3, "B": 4, "C": 5, "dt_bias": "dt_bias"}
    KERNEL = staticmethod(_kernel)

    # -- the call ------------------------------------------------------------------

    def _decoding(self) -> bool:
        return type(self).KERNEL(self) == type(self).RECURRENT_KERNEL

    def _arguments_table(self) -> dict[str, int | str]:
        return self.RECURRENT_ARGUMENTS if self._decoding() else self.CHUNK_ARGUMENTS

    def _arguments(self) -> dict[str, Any]:
        """This call's scan arguments by name, read once per call."""
        call = _this_call(self)
        if "arguments" not in call:
            args, kwargs = getattr(self.source, call["kernel"]).inputs
            named = {name: kwargs.get(at) if isinstance(at, str) else args[at] for name, at in self._arguments_table().items()}
            named["dt_softplus"] = kwargs.get("dt_softplus", False)
            named["dt_limit"] = kwargs.get("dt_limit")
            named["return_final_states"] = kwargs.get("return_final_states", False)
            call["arguments"] = named
        return call["arguments"]

    def _returns_state(self) -> bool:
        return bool(self._arguments()["return_final_states"])

    def _with_seq(self, value: torch.Tensor) -> torch.Tensor:
        """A decode step's tensor with a sequence axis of 1; a prompt's as it is."""
        return value.unsqueeze(1) if self._decoding() else value

    def _without_seq(self, value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(1) if self._decoding() else value

    # -- the values at the scan call -------------------------------------------------

    @EProperty(kernel("inputs"), select=argument("C"), description="C, the queries reading the state, [batch, seq, groups, state_dim]", unavailable=needs_torch_kernels)
    def attention_queries(self, value: torch.Tensor) -> SSDQueries:
        """``C``: what each token reads the state with, ``[batch, seq, groups, state_dim]``, one per group of heads, after the conv and the activation."""
        return self._with_seq(value)

    @attention_queries.postprocess
    def attention_queries(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    @EProperty(kernel("inputs"), select=argument("B"), description="B, the keys writing into the state, [batch, seq, groups, state_dim]", unavailable=needs_torch_kernels)
    def attention_keys(self, value: torch.Tensor) -> SSDKeys:
        """``B``: where each token writes into the state, ``[batch, seq, groups, state_dim]``, one per group of heads."""
        return self._with_seq(value)

    @attention_keys.postprocess
    def attention_keys(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    @EProperty(kernel("inputs"), select=argument("hidden_states"), description="x, the values written into the state, [batch, seq, heads, head_dim]", unavailable=needs_torch_kernels)
    def attention_values(self, value: torch.Tensor) -> SSDValues:
        """``x``: what each token writes into the state, ``[batch, seq, heads, head_dim]``, after the conv and the activation."""
        return self._with_seq(value)

    @attention_values.postprocess
    def attention_values(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    def _betas(self) -> Gates:
        args = self._arguments()
        dt, bias = args["dt"], args["dt_bias"]
        if self._decoding():  # expanded over head_dim for the update: [batch, heads, head_dim], [heads, head_dim]
            dt, bias = dt[..., 0].unsqueeze(1), None if bias is None else bias[..., 0]
        if bias is not None:
            dt = dt + bias.to(dt.dtype)
        if args["dt_softplus"]:
            dt = F.softplus(dt)
        if not self._decoding() and args["dt_limit"] is not None:
            dt = torch.clamp(dt, min=args["dt_limit"][0], max=args["dt_limit"][1])
        return dt

    def _decays(self) -> Gates:
        A = self._arguments()["A"]
        if self._decoding():  # expanded to [heads, head_dim, state_dim] for the update
            A = A[:, 0, 0]
        return A.float() * self._betas().float()

    #: ``dt`` after its bias, softplus and limit: how strongly each token writes into the state.
    betas = DerivedEProperty(
        _betas,
        description="dt, the per-token write strength into the state, [batch, seq, heads]; derived from dt and dt_bias, read-only",
        unavailable=needs_torch_kernels,
    )
    #: ``A * dt``: the log of how much of the state each token keeps.
    decays = DerivedEProperty(
        _decays,
        description="A * dt, the per-token log decay of the state, [batch, seq, heads]; derived, read-only",
        unavailable=needs_torch_kernels,
    )

    @EProperty(kernel("inputs"), select=argument("state"), description="The state entering the layer, [batch, heads, state_dim, head_dim], or None at the start of a prompt (a copy of the cache's buffer)", unavailable=needs_torch_kernels)
    def state_input(self, value: Any) -> State | None:
        """The state this call starts from, key side first: ``None`` on a fresh prompt, the cached state on a decode step.

        A transposed copy: the cache hands the kernel its own buffer and the
        update overwrites it in place, so the live tensor would read as this
        step's *output* by the time the trace ends. Assign to replace what the
        step starts from.
        """
        return value if value is None else value.transpose(-1, -2).clone()

    @state_input.postprocess
    def state_input(self, value: Any) -> Any:
        return value if value is None else value.transpose(-1, -2)

    @EProperty(kernel("output"), select=_head_outputs_select, description="y, the per-head outputs before the gated norm and the output projection, [batch, seq, heads, head_dim]", unavailable=needs_torch_kernels)
    def attention_head_outputs(self, value: torch.Tensor) -> SSDHeadOutputs:
        """``y``: each head's read of the state plus the ``D`` skip, ``[batch, seq, heads, head_dim]``, before the gated norm and ``out_proj``."""
        return self._with_seq(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    @EProperty(_state_output, select=_state_output_select, description="The state leaving the layer, [batch, heads, state_dim, head_dim]", unavailable=needs_torch_kernels)
    def state_output(self, value: torch.Tensor) -> State:
        """The state after this call's last token, ``[batch, heads, state_dim, head_dim]``: what the next decode step starts from."""
        return value.transpose(-1, -2)

    @state_output.postprocess
    def state_output(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(-1, -2)
