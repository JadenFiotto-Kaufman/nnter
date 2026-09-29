"""The descriptors a family's values are made of.

An `EProperty` is nnsight's ``eproperty`` plus availability (`Unavailable`,
``unavailable=``). `SourceEProperty` locates a value at an operation inside a
forward, `RelativeEProperty` at another module named relative to the host,
`DerivedEProperty` computes one from several served values. `branched` and
`per_call` are what a forward that branches needs: a decision made once per
module call and reused by every value read in it."""

from __future__ import annotations

from typing import Any, Callable

from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.intervention.source import SourceEnvoy, SourceNotAvailable


class Unavailable(RuntimeError):
    """A standard value this checkpoint does not have, and why.

    Raised on access, before anything runs, so a wrong assumption fails at the
    line that makes it. `Standard.status` reports the same reasons without
    raising.
    """
    # TODO: make an unavailable value also answer False to ``hasattr``, with the
    # reason intact. Today ``hasattr`` *raises* this, since only AttributeError
    # counts as absence, and nnsight's ``Envoy.__getattr__`` rewrites an
    # AttributeError without the reason; an override of ``__getattr__`` on
    # `Standard` could re-raise `Unavailable` as an AttributeError carrying it.


class EProperty(eproperty):
    """An `eproperty` that can say when it is not available, and why.

    ``unavailable`` is a reason string, or a function of the envoy returning a
    reason or ``None``. A reason makes every read and write raise `Unavailable`
    before the model runs, and shows up in `Standard.status`. It is checked
    on the *instance*, so a checkpoint's config can decide (``attn_implementation``,
    an alibi variant), and a per-layer difference in a hybrid model too.
    """

    def __init__(
        self,
        key: Any = None,
        description: str | None = None,
        unavailable: str | Callable[[Envoy], str | None] | None = None,
    ) -> None:
        self.unavailable = unavailable
        super().__init__(key=key, description=description)

    def __set_name__(self, owner: type, name: str) -> None:
        # A bare marker (`unavailable("...")`) is never called on a stub, so it
        # learns its name from the class body instead.
        if self.name is None:
            self.name = name
        if self.key is None:
            self.key = name

    def reason(self, obj: Envoy) -> str | None:
        """Why this value is not available on ``obj``, or ``None`` when it is."""
        return self.unavailable(obj) if callable(self.unavailable) else self.unavailable

    @property
    def layout(self) -> Any:
        """The value's shape as a ``jaxtyping`` type (``Float[Tensor, "batch seq hidden"]``), or ``None``.

        Read off the return annotation of the function that defines the
        value. ``isinstance(tensor, value.layout)`` checks rank and dtype;
        `dims` names the axes, the same on every family.
        """
        import types
        import typing

        func = self._preprocess
        if func is None:
            return None
        hint = typing.get_type_hints(func, include_extras=True).get("return")
        if isinstance(hint, types.UnionType):  # ``Float[...] | None``
            hint = next((arg for arg in typing.get_args(hint) if arg is not type(None)), None)
        return hint if hasattr(hint, "dim_str") else None

    @property
    def dims(self) -> tuple[str, ...] | None:
        """The axis names of `layout`: ``("batch", "seq", "hidden")``."""
        layout = self.layout
        return tuple(layout.dim_str.split()) if layout is not None else None

    def _check(self, obj: Envoy) -> None:
        try:
            reason = self.reason(obj)
        except AttributeError as error:
            # Never let this surface as an AttributeError: a descriptor's
            # AttributeError falls through to Envoy.__getattr__ and comes back
            # as "no attribute 'name'", hiding the predicate's own bug.
            raise RuntimeError(
                f"the availability check of {obj.path}.{self.name} failed: {error}"
            ) from error
        if reason:
            raise Unavailable(f"{obj.path}.{self.name} is not available: {reason}")

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        return super().__get__(obj, owner)

    def __set__(self, obj: Envoy, value: Any) -> None:
        self._check(obj)
        super().__set__(obj, value)


def unavailable(reason: str) -> EProperty:
    """A value a family does not have: assign it in the class body in place of the inherited one.

    ``attention_probabilities = unavailable("no softmax: the attention is linear")``
    keeps the name in the tree and in the repr, with the reason, and makes any
    access raise `Unavailable` with it.
    """
    return EProperty(description=f"Unavailable: {reason}", unavailable=reason)


class SourceEProperty(EProperty):
    """An `eproperty` over an operation inside the module's forward.

    Args:
        op: The operation's dotted path under the module's ``.source``, with
            ``.source.`` between a call and an operation inside it:
            ``"attention_interface_1.source.nn_functional_softmax_0"``. Or a
            function of the envoy returning that path, for a forward that
            branches: it runs at read time, inside the trace, so it can read
            the forward's own branch variable (a binding is an operation
            too) and name the op that fires on this call.
        attribute: Which of the operation's served values this is
            (``"output"``, ``"input"``). The location is
            ``{path}.source.{op}.{attribute}``, exactly what
            ``envoy.source.<op>.<attribute>`` reads in a block.
        description: Shown in the model's repr, like any eproperty's.

    Every read and write first walks ``obj.source`` the way a user would in
    the block, so the operation is instrumented for the current run, then
    reads or writes the operation's own ``.output`` / ``.input`` descriptor,
    with the repacking those already do (assigning ``.input`` replaces the
    first argument and keeps the rest). An operation that is not there raises
    `SourceNotAvailable` naming what is, rather than the `AttributeError` a
    descriptor would otherwise swallow into "no attribute".

    ``select`` picks one element of the served value: with ``attribute="inputs"``
    an int is a positional argument and a str a keyword; with ``"output"`` an
    int indexes the returned tuple. A write repacks the element into the
    current value and writes that back, so an assignment to one argument of
    a call replaces just that argument. The element is the object the call
    holds, so in-place edits reach the model without a transform (nnsight's
    ``eproperty.transform`` is not wired here: this descriptor reads through
    the operation's own descriptors rather than ``eproperty.__get__``).
    """

    def __init__(
        self,
        op: str | Callable[[Envoy], str],
        attribute: str = "output",
        description: str | None = None,
        unavailable: str | Callable[[Envoy], str | None] | None = None,
        select: int | str | None = None,
    ) -> None:
        name = op if isinstance(op, str) else f"<{op.__name__}>"
        super().__init__(key=f"source.{name}.{attribute}", description=description, unavailable=unavailable)
        self.op = op
        self.attribute = attribute
        self.select = select

    def _pick(self, value: Any) -> Any:
        if self.select is None:
            return value
        if self.attribute == "inputs":
            args, kwargs = value
            return kwargs[self.select] if isinstance(self.select, str) else args[self.select]
        return value[self.select]

    def _put(self, current: Any, element: Any) -> Any:
        if self.select is None:
            return element
        if self.attribute == "inputs":
            args, kwargs = current
            if isinstance(self.select, str):
                return args, {**kwargs, self.select: element}
            args = list(args)
            args[self.select] = element
            return tuple(args), kwargs
        current = list(current)
        current[self.select] = element
        return tuple(current)

    def _drill(self, obj: Envoy) -> SourceEnvoy:
        op = self.op if isinstance(self.op, str) else self.op(obj)
        *calls, name = op.split(".source.")
        source = obj.source
        try:
            for call in calls:
                source = getattr(source, call).source if calls and self._is_drilled(obj, source, call) else self._drill_relaxed(source, call)
            return getattr(source, name)
        except AttributeError as error:
            raise SourceNotAvailable(
                f"{obj.path}.{self.name} reads operation {op!r} under .source, "
                f"which this run does not have: {error}. The forward took a path "
                f"this family's toolkit does not expect."
            ) from None

    @staticmethod
    def _is_drilled(obj: Envoy, source: Any, call: str) -> bool:
        return getattr(source, call).path in obj.interleaver.sourced

    @staticmethod
    def _drill_relaxed(source: Any, call: str) -> Any:
        """Drill into ``call`` the first time this run with the mediator relaxed.

        Drilling resolves the callee from the live call, a served read of the
        call's ``.fn``. A read pinned by ``tracer.iter`` to a token index would
        ask for the call's *n-th occurrence*, which a call that fires once per
        forward never reaches; the call is the one in flight, so the read is
        made relaxed and the pin restored for the value read that follows.
        """
        from nnsight.intervention.interleaver import Mediator

        mediator = Mediator.current(call)
        pinned, mediator.iteration = mediator.iteration, None
        try:
            return getattr(source, call).source
        finally:
            mediator.iteration = pinned

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        value = self._pick(getattr(self._drill(obj), self.attribute))
        return self._preprocess(obj, value) if self._preprocess is not None else value

    def __set__(self, obj: Envoy, value: Any) -> None:
        self._check(obj)
        if self._postprocess is not None:
            value = self._postprocess(obj, value)
        op = self._drill(obj)
        if self.select is not None:
            value = self._put(getattr(op, self.attribute), value)
        setattr(op, self.attribute, value)


class RelativeEProperty(EProperty):
    """An `eproperty` served at another module, named relative to this envoy.

    ``key`` is ``"<path>.<attribute>"``. A path resolves from this envoy the
    way attribute access does, aliases included (``"embed_tokens.output"`` on
    the root reaches GPT-2's ``transformer.wte``); a leading ``../`` steps to
    the parent first, by native name (``"../post_attention_layernorm.output"``
    on an attention module reaches its sibling norm). For a value that belongs
    to this module in meaning but is produced elsewhere in the tree: what a
    sandwich block's attention adds to the residual stream is the post-attention
    norm's output, so that family's `Attention.attention_output` points there.
    """

    def _location(self, obj: Envoy) -> str:
        path, _, attribute = self.key.rpartition(".")
        if path.startswith("../"):
            parent = obj.path.rsplit(".", 1)[0]
            return f"{parent}.{path.removeprefix('../')}.{attribute}"
        return f"{obj.get(path).path}.{attribute}"

def branched(variable: str, ops: dict[Any, str]) -> Callable[[Envoy], str]:
    """An ``op`` for `SourceEProperty` that a forward's own branch variable picks.

    ``variable`` names a binding the forward makes before it branches (a
    binding is an operation, so its value is served like any other), and
    ``ops`` maps that value to the op that fires on that branch.

    The variable is read once per step: the model serves a location once, so
    a second value read in the same call must not ask for it again after the
    model has moved on. The choice is cached on the envoy against the worker's
    mediator and its step: inside ``tracer.iter`` a step body starts pinned to
    its step and relaxes after its first read, and a plain trace stays at 0,
    so a pinned step different from the cached one is a new call, a relaxed
    one is the same call, and another mediator is another run. Reading the
    variable as the step's first, pinned read also leaves the kernel read
    sequential, which is what makes an op that never fires on step 0 resolve
    on later steps.
    """

    def choose(envoy: Envoy) -> str:
        return per_call(envoy, f"branch:{variable}", lambda: ops[getattr(envoy.source, variable).output])

    choose.__name__ = f"branched({variable})"
    return choose


def per_call(envoy: Envoy, key: str, compute: Callable[[], Any]) -> Any:
    """``compute()`` once per module call, cached on the envoy under ``key``.

    For a served value several reads in one call depend on, when the model
    serves it once: the branch a forward takes, the sequence length of a call.
    A call is told apart by the worker's mediator and its step (see
    `branched`): a pinned step different from the cached one is a new call, a
    relaxed one the same call, another mediator another run.
    """
    from nnsight.intervention.interleaver import Mediator

    mediator = Mediator.current(key)
    step = mediator.iteration
    cache = envoy.__dict__.setdefault("_per_call", {})
    cached = cache.get(key)
    if cached is None or cached[0] is not mediator or (step is not None and step != cached[1]):
        cache[key] = cached = (mediator, step, compute())
    return cached[2]


class DerivedEProperty(EProperty):
    """An `EProperty` computed from other served values rather than read at one location.

    ``compute(envoy)`` runs at read time inside the trace and may read any
    number of values; the result is read-only (assign through whatever
    method the host offers). Listed in the repr like any value.
    """

    def __init__(self, compute: Callable[[Envoy], Any], description: str | None = None, unavailable: Any = None) -> None:
        super().__init__(key=f"<{compute.__name__}>", description=description, unavailable=unavailable)
        self.compute = compute
        self._preprocess = compute  # what `layout` reads the annotation from; never called as a preprocess

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        return self.compute(obj)  # not `_preprocess(obj, value)`: there is no served value

    def __set__(self, obj: Envoy, value: Any) -> None:
        raise AttributeError(f"{self.name} is derived and read-only")

def at_occurrence(t: int):
    """The ``for step in tracer.iter[t]`` stretch, for one occurrence of a location inside a call."""
    from nnsight.intervention.iterator import Iterations

    return Iterations()[t : t + 1]
