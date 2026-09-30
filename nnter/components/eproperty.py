"""The descriptors a family's values are made of.

An `EProperty` is nnsight's ``eproperty`` with two additions: availability
(`Unavailable`, ``unavailable=``) and a *path* for a key, so one descriptor
serves a value wherever it lives: on the host module, on another module named
relative to it, or at an operation inside a forward. `DerivedEProperty`
computes one from several served values. `branched` and `per_call` are what a
forward that branches needs: a decision made once per module call and reused
by every value read in it."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.intervention.interleaver import Mediator
from nnsight.intervention.source import SourceEnvoy, SourceNotAvailable
from nnsight.intervention.util import first_input, replace_first_input


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
    """An ``eproperty`` whose key is a path from the host, and which can say when it is unavailable.

    Args:
        key: Where the value lives, relative to the host envoy: dotted segments
            ending in ``output``, ``input`` or ``inputs``. ``"output"`` is the
            host's own output. A leading ``"../"`` (repeatable) steps to the
            parent module by native name, another name to a child module
            (aliases included) or, under a ``source`` segment, to an
            operation: ``"source"`` drills into the current module's or
            operation's forward, instrumenting it for this run, so
            ``"source.attention_interface_1.inputs"`` is the shared attention
            call's arguments. Above the host the location is the path's
            string, so ``"../source.hidden_states_view_0.output"`` is an
            operation of the parent block's own forward, which the family
            instruments at build (`Standard.sourced`); a call inside that
            forward cannot be drilled from a child. A function of the host returning such a path, for a
            forward that branches: it runs at read time, inside the trace, so it
            can read the forward's own branch variable (`branched`) or the
            config (a family's ``by_alibi``). ``None`` means the attribute's
            name, for a bare marker (`unavailable`).
        description: Shown in the model's repr, like any eproperty's.
        unavailable: A reason string, or a function of the envoy returning a
            reason or ``None``. A reason makes every read and write raise
            `Unavailable` before the model runs, and shows up in
            `Standard.status`. It is checked on the *instance*, so a
            checkpoint's config can decide (``attn_implementation``, an alibi
            variant), and a per-layer difference in a hybrid model too.
        select: One element of the served value: with ``inputs`` an int is a
            positional argument and a str a keyword; with ``output`` an int
            indexes the returned tuple. A write repacks the element into the
            current value, so assigning one argument of a call replaces just
            that argument. ``input`` is the call's first argument, ``inputs``
            with the first element selected. A function of the host returning
            one of these (or ``None``, the whole value) for a key that branches
            onto calls whose arguments sit at different positions (Mamba's
            prompt and decode kernels); it runs at read time, after the key.

    The location is served by nnsight the way any eproperty's is, whatever the
    path: a module's output, a sibling norm's, or an operation's arguments,
    with ``preprocess``, ``postprocess`` and ``transform`` all available. An
    operation inside a called function only exists once someone has drilled
    into that call in the *current* run (the interleaver resolves the callee
    from the live value and clears what it built at the start of every run),
    so a path through ``source`` is walked before every read or write. An
    operation that is not there raises `SourceNotAvailable` naming what is,
    rather than the `AttributeError` a descriptor would otherwise swallow into
    "no attribute".
    """

    def __init__(
        self,
        key: str | Callable[[Envoy], str] | None = None,
        description: str | None = None,
        unavailable: str | Callable[[Envoy], str | None] | None = None,
        select: int | str | Callable[[Envoy], int | str | None] | None = None,
    ) -> None:
        self.locate = key if callable(key) else None
        self.unavailable = unavailable
        self.select = select
        super().__init__(key=f"<{key.__name__}>" if callable(key) else key, description=description)

    def __set_name__(self, owner: type, name: str) -> None:
        # A bare marker (`unavailable("...")`) is never called on a stub, so it
        # learns its name from the class body instead.
        if self.name is None:
            self.name = name
        if self.key is None:
            self.key = name

    # -- availability -----------------------------------------------------------

    def reason(self, obj: Envoy) -> str | None:
        """Why this value is not available on ``obj``, or ``None`` when it is."""
        return self.unavailable(obj) if callable(self.unavailable) else self.unavailable

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

    # -- the layout -------------------------------------------------------------

    @property
    def layout(self) -> Any:
        """The value's shape as a ``jaxtyping`` type (``Residual``, ``Pattern``, ...), or ``None``.

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
        if isinstance(hint, types.UnionType):  # ``State | None``
            hint = next((arg for arg in typing.get_args(hint) if arg is not type(None)), None)
        return hint if hasattr(hint, "dim_str") else None

    @property
    def dims(self) -> tuple[str, ...] | None:
        """The axis names of `layout`: ``("batch", "seq", "hidden")``."""
        layout = self.layout
        return tuple(layout.dim_str.split()) if layout is not None else None

    # -- the location -----------------------------------------------------------

    def path(self, obj: Envoy) -> str:
        """The key for ``obj``: the path itself, or what the key function returns for it."""
        return self.locate(obj) if self.locate is not None else self.key

    def inside_forward(self, obj: Envoy | None = None) -> bool:
        """Whether the value is an operation inside a forward (a ``source`` segment on its path)."""
        key = (self.path(obj) if obj is not None else self.key) or ""
        return self.locate is not None or "source" in key.lstrip("./").split(".")

    def _location(self, obj: Envoy) -> str:
        return self._resolve(obj, self.path(obj))

    def _resolve(self, obj: Envoy, key: str) -> str:
        """The served location ``key`` names from ``obj``, walking (and drilling) the path."""
        up = 0
        while key.startswith("../"):
            up, key = up + 1, key[3:]
        *walk, attribute = key.split(".")
        attribute = "input" if attribute in ("input", "inputs") else "output"
        if up:
            # Above the host the path is arithmetic on names: an envoy knows its
            # own path but not its parent, and the parent's forward, when the
            # path goes into it, is instrumented already (`Standard.sourced`),
            # so the location is served by its string. One level of ``source``
            # is what that gives; a call inside the parent's forward would need
            # the parent drilled, which only an envoy can do.
            parts = obj.path.split(".")[:-up]
            if walk.count("source") > 1:
                raise ValueError(
                    f"{obj.path}.{self.name}: {key!r} drills into a call inside the parent's forward; "
                    "a path above the host reaches the parent's own operations only"
                )
            return ".".join([*parts, *walk, attribute])
        node: Any = obj
        try:
            for segment in walk:
                if segment == "source":
                    node = _drill(obj, node)
                elif isinstance(node, Envoy):
                    node = node.get(segment)
                else:
                    node = getattr(node, segment)
        except AttributeError as error:
            raise SourceNotAvailable(
                f"{obj.path}.{self.name} reads {key!r}, which this run does not have: "
                f"{error}. The forward took a path this family's toolkit does not expect."
            ) from None
        return f"{node.path}.{attribute}"

    # -- select -----------------------------------------------------------------

    def _selector(self, obj: Envoy) -> int | str | None:
        """The element this access selects: `select`, or what it returns for ``obj``."""
        return self.select(obj) if callable(self.select) else self.select

    def _pick(self, attribute: str, value: Any, select: int | str | None) -> Any:
        if attribute == "input":
            return first_input(*value)
        if select is None:
            return value
        if attribute == "inputs":
            args, kwargs = value
            return kwargs[select] if isinstance(select, str) else args[select]
        return value[select]

    def _put(self, attribute: str, current: Any, element: Any, select: int | str | None) -> Any:
        if attribute == "input":
            return replace_first_input(*current, element)
        if select is None:
            return element
        if attribute == "inputs":
            args, kwargs = current
            if isinstance(select, str):
                return args, {**kwargs, select: element}
            args = list(args)
            args[select] = element
            return tuple(args), kwargs
        current = list(current)
        current[select] = element
        return tuple(current)

    # -- read and write -----------------------------------------------------------
    # The key is computed once per access: a key function may read a served
    # value pinned to the current step (a forward's branch variable), and a
    # second evaluation after the read would run with the pin relaxed.

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        key = self.path(obj)
        location = self._resolve(obj, key)
        raw = Mediator.value(location)
        value = self._pick(key.rsplit(".", 1)[-1], raw, self._selector(obj))
        if self._preprocess is not None:
            value = self._preprocess(obj, value)
        if self._transform is not None:
            # Bound now so the user's in-place edits on the returned view are
            # visible when the mediator fires it after this read (nnsight's
            # eproperty does the same); the raw served value rides along for a
            # write-back that has to rebuild a container around the view.
            Mediator.current(location).transform = partial(self._transform, obj, value, raw)
        return value

    def __set__(self, obj: Envoy, value: Any) -> None:
        self._check(obj)
        if self._postprocess is not None:
            value = self._postprocess(obj, value)
        key = self.path(obj)
        location = self._resolve(obj, key)
        attribute = key.rsplit(".", 1)[-1]
        select = self._selector(obj)
        if select is not None or attribute == "input":
            value = self._put(attribute, Mediator.value(location), value, select)
        Mediator.swap(location, value)


def _drill(obj: Envoy, node: Any) -> Any:
    """``node.source``: the module's forward instrumented, or an operation's callee drilled into.

    Drilling into a call resolves the callee from the live call, a served
    read of the call's ``.fn``. A read pinned by ``tracer.iter`` to a token
    index would ask for the call's *n-th occurrence*, which a call that fires
    once per forward never reaches; the call is the one in flight, so the
    first drill of a run is made with the mediator relaxed and the pin
    restored for the value read that follows.
    """
    if not isinstance(node, SourceEnvoy) or node.path in obj.interleaver.sourced:
        return node.source
    mediator = Mediator.current(node.path)
    pinned, mediator.iteration = mediator.iteration, None
    try:
        return node.source
    finally:
        mediator.iteration = pinned


def unavailable(reason: str) -> EProperty:
    """A value a family does not have: assign it in the class body in place of the inherited one.

    ``attention_probabilities = unavailable("no softmax: the attention is linear")``
    keeps the name in the tree and in the repr, with the reason, and makes any
    access raise `Unavailable` with it.
    """
    return EProperty(description=f"Unavailable: {reason}", unavailable=reason)


def branched(variable: str, ops: dict[Any, str]) -> Callable[[Envoy], str]:
    """An op name a forward's own branch variable picks, for a key function (see `EProperty`).

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
    `branched`): a pinned step different from the cached one is a new call,
    another mediator another run. A relaxed read counts as the step of the
    envoy's last pinned one, so a value first computed after the step's
    first read (a DeltaNet's per-token offset, read after ``state_input``)
    is not the previous step's.
    """
    from nnsight.intervention.interleaver import Mediator

    mediator = Mediator.current(key)
    step = mediator.iteration
    cache = envoy.__dict__.setdefault("_per_call", {})
    if step is not None:
        cache[None] = (mediator, step)
    elif cache.get(None, (None,))[0] is mediator:
        step = cache[None][1]
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
