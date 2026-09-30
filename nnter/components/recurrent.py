"""`RecurrentMixer`: how a recurrent mixer's values are reached, whatever the values are.

A recurrent mixer (a gated DeltaNet, a state-space layer) runs its sequence
through a kernel function the modeling module calls: one kernel for a prompt,
another for a decode step of ``generate``, picked by a variable the forward
binds before it branches. Its values are the arguments and results of that
call. What differs between mixers is which arguments mean what; what they
share is how the call is found, which kernel names must be transformers'
pure-torch ones for there to be a call to read inside, how the family's
kernels are rebound (`route_kernels`), and, when the token-by-token kernel
materializes it, the state after every token. That shared part is
`RecurrentMixer`; a subclass (`LinearAttention`, `StateSpace`) names its
kernels in four class constants and declares its values.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import SourceEnvoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty, Unavailable, branched, per_call
from .layer import Residual
from .standard import Standard, first_tensor, rewrap

#: The recurrent state alone, one ``key_dim`` by ``value_dim`` matrix per head, and stacked along the tokens.
State = Float[Tensor, "batch heads key_dim value_dim"]
States = Float[Tensor, "batch seq heads key_dim value_dim"]


def _name(op: str) -> str:
    """The module-level function an op calls: ``torch_chunk_gated_delta_rule_0`` -> ``torch_chunk_gated_delta_rule``."""
    return op.rsplit("_", 1)[0]


def _dispatch(bound: Any) -> dict[str, Any]:
    """The closure of transformers' kernel dispatcher (``use_kernel_func_from_hub_with_fallback``), or ``{}``.

    It holds ``torch_function``, the pure-torch kernel, and ``implementation``,
    the one the dispatcher calls: an optimized kernel when one is installed,
    ``torch_function`` otherwise.
    """
    return inspect.getclosurevars(bound).nonlocals if getattr(bound, "__closure__", None) else {}


def _torch_function(bound: Any) -> Callable:
    """The pure-torch kernel behind a module-level name: the dispatcher's ``torch_function``, or the name's own binding."""
    return _dispatch(bound).get("torch_function", bound)


def _modeling_module(envoy: Envoy):
    return sys.modules[type(envoy._module).__module__]


def _mixer(family) -> tuple[Any, type[RecurrentMixer]]:
    """The transformers modeling module a family's recurrent mixer lives in, and the mixer's envoy class."""
    from types import ModuleType

    if isinstance(family, ModuleType) and not hasattr(family, "ENVOYS"):
        # Already the modeling module: the mixer is the envoy a loaded family keys on one of its
        # classes, else a mixer class whose kernels it defines.
        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        mixers = list(subclasses(RecurrentMixer))
        for cls in mixers:
            envoys = getattr(sys.modules.get(cls.__module__), "ENVOYS", {})
            if any(envoy is cls and key.__module__ == family.__name__ for key, envoy in envoys.items()):
                return family, cls
        for cls in mixers:
            if cls.CHUNK_KERNEL and hasattr(family, _name(cls.CHUNK_KERNEL)):
                return family, cls
        raise ValueError(f"{family.__name__} defines no recurrent mixer's kernels")
    found = next(
        ((module, envoy) for module, envoy in family.ENVOYS.items() if issubclass(envoy, RecurrentMixer)), None
    )
    if found is None:
        raise ValueError(f"{family.__name__} has no recurrent mixer to route")
    module, envoy = found
    return sys.modules[module.__module__], envoy


def route_kernels(family, kernel: str = "torch") -> None:
    """Bind a family's recurrent kernels, process-wide.

    ``"torch"`` binds the prompt's and the decode step's kernel names in the
    family's modeling module to transformers' pure-torch kernels, the
    functions the dispatcher falls back to: the slower path, and the only one
    with Python source to read values inside. On a mixer whose token-by-token
    kernel materializes the state after every token (``STATE_OP`` set), both
    names are bound to that kernel, so a prompt runs through it too and
    `RecurrentMixer.state`, `states`, `state_after` and `set_state_after`
    read and write the state at any position. ``"default"`` restores what the
    module bound at import. Like installing a kernel, it applies to every
    model of the family; call it before tracing a layer, since a forward
    ``.source`` has already instrumented keeps the binding it was compiled
    with. ``family`` is the family module (``model.family``, or
    ``nnter.families.qwen3_5_text``) or its modeling module.
    """
    module, mixer = _mixer(family)
    names = (_name(mixer.CHUNK_KERNEL), _name(mixer.RECURRENT_KERNEL))
    originals = module.__dict__.setdefault("_nnter_kernels", {})
    for name in names:
        originals.setdefault(name, getattr(module, name))
    if kernel == "torch":
        if mixer.STATE_OP is not None:
            loop = _torch_function(originals[_name(mixer.RECURRENT_KERNEL)])
            bindings = dict.fromkeys(names, loop)
        else:
            bindings = {name: _torch_function(originals[name]) for name in names}
    elif kernel == "default":
        bindings = {name: originals[name] for name in names}
    else:
        raise ValueError(f"kernel must be 'torch' or 'default', not {kernel!r}")
    for name, function in bindings.items():
        setattr(module, name, function)


def route_delta_rule(family, kernel: str = "recurrent") -> None:
    """`route_kernels` for a gated DeltaNet, in its own words.

    ``"recurrent"`` is ``route_kernels(family, "torch")``: the family's
    prompts run through transformers' token-by-token gated delta rule, which
    materializes the state after every token. ``"chunked"`` is
    ``route_kernels(family, "default")``.
    """
    routes = {"recurrent": "torch", "chunked": "default"}
    if kernel not in routes:
        raise ValueError(f"delta_rule must be 'recurrent' or 'chunked', not {kernel!r}")
    route_kernels(family, routes[kernel])


def needs_torch_kernels(envoy: Envoy) -> str | None:
    """Why a recurrent mixer's kernel values are unavailable: the kernel has no Python source to read inside."""
    cls = type(envoy)
    module = _modeling_module(envoy)
    for op in (cls.CHUNK_KERNEL, cls.RECURRENT_KERNEL):
        closure = _dispatch(getattr(module, _name(op), None))
        implementation = closure.get("implementation")
        if implementation is not None and implementation is not closure["torch_function"]:
            package = getattr(implementation, "__module__", None) or "an installed package"
            return (
                f"read inside transformers' pure-torch {_name(op)}, but this process dispatches it "
                f"to an optimized kernel ({package.split('.')[0]}) with no Python source; "
                "uninstall it, or call nnter.route_kernels(model.family, 'torch'), to read these"
            )
    return None


def needs_recurrent_routing(envoy: Envoy) -> str | None:
    """Why the per-token state is unavailable: only the token-by-token kernel materializes it."""
    cls = type(envoy)
    if cls.STATE_OP is None:
        return "this mixer's kernels do not materialize the state per token"
    reason = needs_torch_kernels(envoy)
    if reason:
        return reason
    module = _modeling_module(envoy)
    loop = _torch_function(getattr(module, _name(cls.RECURRENT_KERNEL)))
    if getattr(module, _name(cls.CHUNK_KERNEL)) is not loop:
        return (
            "the state after each token is materialized only by the token-by-token kernel; "
            "the chunked kernel a prompt runs through carries it between chunks. Call "
            "nnter.route_kernels(model.family, 'torch') before tracing this layer "
            "(slower, like attn_implementation='eager')"
        )
    return None


def at_occurrence(t: int):
    """The ``for step in tracer.iter[t]`` stretch, for one occurrence of a location inside a call."""
    from nnsight.intervention.iterator import Iterations

    return Iterations()[t : t + 1]


def kernel(attribute: str) -> Callable[[Envoy], str]:
    """A key at whichever kernel fires on this call: ``source.<KERNEL>.<attribute>``."""

    def locate(envoy: Envoy) -> str:
        return f"source.{type(envoy).KERNEL(envoy)}.{attribute}"

    locate.__name__ = f"kernel.{attribute}"
    return locate


class RecurrentMixer(Standard):
    """A sequence mixer with a recurrent state, read at the kernel call its forward makes.

    The mechanism every recurrent mixer shares, apart from what its values
    are. A subclass names its kernels in four class constants and declares
    its values at ``kernel("inputs")`` / ``kernel("output")``:

    * `BRANCH`: the binding the forward makes before it picks a kernel,
      ``False`` on a prompt and ``True`` on a decode step.
    * `CHUNK_KERNEL`: the call a prompt runs through.
    * `RECURRENT_KERNEL`: the call each decode step of ``generate`` runs
      through.
    * `STATE_OP`: inside the token-by-token kernel, the binding of the state
      after each token's update; ``None`` when the mixer's kernels do not
      materialize it.

    `KERNEL` reads `BRANCH` once per call and names the call that fires on
    this step, so the same values work in a ``trace`` and at every step of
    ``tracer.iter``. The kernels have to be transformers' pure-torch ones
    for there to be a call to read inside (`needs_torch_kernels`).

    The state *after every token* of a prompt exists only in the
    token-by-token kernel: a chunked kernel carries the state between chunks.
    Like eager attention, that is the user's choice:
    ``nnter.route_kernels(model.family, "torch")`` routes the family's
    prompts through the token-by-token kernel, and then `state`, `states`,
    `state_after` and `set_state_after` read and write the state at any
    position; without it, or on a mixer with no `STATE_OP`, reading one
    raises `Unavailable` with the reason and `status` reports it.
    """

    #: The binding the forward makes before it branches: ``False`` on a prompt, ``True`` on a decode step.
    BRANCH = "use_precomputed_states_0"
    #: The call a prompt runs through.
    CHUNK_KERNEL: str | None = None
    #: The call each decode step runs through.
    RECURRENT_KERNEL: str | None = None
    #: Inside the token-by-token kernel, the binding of the state after each token's update, or ``None``.
    STATE_OP: str | None = None
    #: Whichever kernel fires on this call, chosen by `BRANCH`; built from the constants.
    KERNEL: Callable[[Envoy], str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = vars(cls)
        if "KERNEL" not in declared and any(name in declared for name in ("BRANCH", "CHUNK_KERNEL", "RECURRENT_KERNEL")):
            if cls.CHUNK_KERNEL and cls.RECURRENT_KERNEL:
                cls.KERNEL = staticmethod(branched(cls.BRANCH, {False: cls.CHUNK_KERNEL, True: cls.RECURRENT_KERNEL}))

    @EProperty(key="output", description="What the mixer adds to the residual stream")
    def attention_output(self, value: Any) -> Residual:
        """The mixer's contribution to the residual stream, a tensor."""
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)

    # -- the state at every token: the token-by-token kernel only -------------------

    @staticmethod
    def _token_state_op(envoy: Envoy) -> str:
        # Unpinned, or pinned to index 0, a read decides the kernel like every
        # other value (once per call, cached), so a call-level value read after
        # a token loop still finds the branch decided. Pinned to a later token
        # index it cannot read the branch variable — that fires once per call —
        # so it takes the prompt's kernel, the one a token loop walks.
        from nnsight.intervention.interleaver import Mediator

        cls = type(envoy)
        step = Mediator.current("state").iteration
        kernel = cls.KERNEL(envoy) if step in (None, 0) else cls.CHUNK_KERNEL
        return f"source.{kernel}.source.{cls.STATE_OP}.output"

    @EProperty(_token_state_op, description="The recurrent state after one token of the prompt; iterate it with tracer.iter; needs route_kernels(family, 'torch')", unavailable=needs_recurrent_routing)
    def state(self, value: torch.Tensor) -> State:
        """The state after a token of the prompt: one occurrence per token.

        A location inside the token-by-token kernel, so it takes nnsight's
        own iteration: ``for t in tracer.iter[:]: mix.state`` reads the state
        after every token, ``tracer.iter[4]`` the one after token 4, and an
        assignment there is a write the following tokens continue from.
        Outside any ``tracer.iter`` a read is the state after token 0. It
        lives on the prompt's kernel call: under ``generate`` walk it in an
        inner ``tracer.iter`` on step 0, and take a decode step's state from
        the call's output, one token per step. Needs ``route_kernels(family, "torch")``.
        """
        return value

    def _seq(self) -> int:
        """This call's number of tokens, read off the kernel call (the queries' sequence axis on a DeltaNet)."""
        return self.attention_queries.shape[1]

    def _call(self) -> tuple[int, int]:
        """This call's sequence length and the occurrence its first token is, decided once per call.

        The kernel's arguments are served once per call: reading them parks
        the worker at the call's start, which is the one moment the state
        op's occurrence count is the number of tokens *earlier* calls put
        through it. Occurrences are counted per location over the whole run,
        so a decode step's single token is not occurrence 0 once an earlier
        step has fired the recurrent kernel's op. Whichever per-token read
        comes first in the call takes both numbers; the rest reuse them.
        """
        from nnsight.intervention.interleaver import Mediator

        def compute():
            seq = self._seq()
            op = getattr(getattr(self.source, type(self).KERNEL(self)).source, self.STATE_OP)
            location = f"{op.path}.output"
            return seq, Mediator.current(location).occurrence(location)

        return per_call(self, "call", compute)

    def _token_op(self) -> tuple[SourceEnvoy, int]:
        """This call's state-update op and the occurrence its first token is (see `_call`)."""
        _, first = self._call()
        op = getattr(getattr(self.source, type(self).KERNEL(self)).source, self.STATE_OP)
        return op, first

    def _states(self) -> States:
        # Through whichever kernel call fires on this step, so a decode step's
        # one-token call answers too; `state` alone is the prompt's location.
        seq, _ = self._call()
        op, first = self._token_op()
        states = []
        for t in range(seq):
            for _ in at_occurrence(first + t):
                states.append(op.output)
        return torch.stack(states, dim=1)

    #: The state after each token, stacked on a sequence axis: one occurrence of the token-by-token
    #: kernel's state update per token, so it needs ``route_kernels(family, "torch")``. The last is the
    #: state the call leaves. Read-only, a stack of copies: `set_state_after` writes one position.
    states = DerivedEProperty(
        _states,
        description="The recurrent state after every token of this call; needs route_kernels(family, 'torch')",
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
