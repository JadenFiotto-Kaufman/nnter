---
title: EProperty Internals
one_liner: How nnter's four descriptors sit on nnsight's eproperty — availability, drilling into .source per run, relative locations, derived values, and the per-call cache a branching forward needs.
tags: [developing, internals, eproperty, source, descriptors]
related: [docs/developing/architecture.md, docs/developing/linear-attention-internals.md, docs/developing/gotchas.md, docs/usage/availability.md]
sources: [nnter/components/eproperty.py, nnter/components/attention.py, nnter/components/standard.py, nnter/families/falcon.py, nnsight src/nnsight/intervention/eproperty.py, nnsight src/nnsight/intervention/source.py, nnsight src/nnsight/intervention/interleaver.py, nnsight src/nnsight/intervention/iterator.py]
---

# EProperty Internals

## What this is for

Every standard value is a descriptor from `nnter/components/eproperty.py`,
and every one of them is nnsight's `eproperty` (a `property` subclass over a
location string) plus one thing nnsight does not have: an answer, before
anything runs, to "does this checkpoint have this value, and why not". This
page is the contract of each descriptor and the nnsight fact it depends on,
with the line that establishes it on each side. Read nnsight
`docs/developing/extending-envoy.md` first if `eproperty`, location, and
`Mediator` are new words.

## Canonical pattern

A value of your own, located inside the shared attention forward, on a
Llama-family model (run on `hf-internal-testing/tiny-random-LlamaForCausalLM`):

```python
import torch
from jaxtyping import Float
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention   # after `import nnter`

import nnter
from nnter import StandardizedTransformer
from nnter.components import INTERFACE, SourceEProperty, interface_reason
from nnter.families import llama


class Attention(llama.Attention):
    @SourceEProperty(f"{INTERFACE}.source.repeat_kv_0", description="The keys after repeat_kv, [batch, heads, seq, head_dim]", unavailable=interface_reason)
    def expanded_keys(self, value: torch.Tensor) -> Float[Tensor, "batch heads seq head_dim"]:
        return value


model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True, attn_implementation="eager", envoys={LlamaAttention: Attention})
attn = model.layers[0].self_attn

Attention.expanded_keys.key      # 'source.attention_interface_1.source.repeat_kv_0.output'
Attention.expanded_keys.dims     # ('batch', 'heads', 'seq', 'head_dim')
"(expanded_keys): The keys after repeat_kv" in repr(attn)   # True

with model.trace("Hello world there"):
    keys = attn.attention_keys.save()       # [batch, kv_heads, seq, head_dim], read first: repeat_kv fires later
    expanded = attn.expanded_keys.save()    # [batch, heads, seq, head_dim]
```

## The nnsight side, in five facts

1. **An `eproperty` is a `property` over `"{obj.path}.{key}"`.** Reading
   parks the worker with `Mediator.value(location)`, runs the decorated stub
   as the *preprocess* on the served value, and returns the result; writing
   runs `postprocess` and `Mediator.swap` (nnsight `eproperty.py:168-190`).
   `key` defaults to the stub's name (`:144-151`), and a `description` is
   only what the repr prints (`envoy.py:1082-1095`).
2. **A getter's `AttributeError` is swallowed.** A `property` raising
   `AttributeError` falls through to `Envoy.__getattr__` (`envoy.py:802-819`),
   which reports `'X' object (nor its module) has attribute 'name'`; the real
   error is lost (`eproperty.py:68-72`).
3. **`transform` is the write-back of a reshaping preprocess.** `__get__`
   binds `partial(transform, obj, view, raw)` onto the current mediator
   (`eproperty.py:178-184`); `Mediator.handle` fires it once, after the
   worker's read on that location, and splices the result in like a swap
   (`interleaver.py:487-497`). Its signature is `(self, view, raw)`: the
   edited view and the value as served (`eproperty.py:158-166`).
4. **A location can be visited many times in one run**, and a worker asks
   for one *occurrence* of it (`Pending.iteration`, `interleaver.py:85-96`).
   `Mediator.iteration` is the occurrence the worker wants: `0` with no
   `tracer.iter`, an int a `tracer.iter[n]` pins, `None` when *relaxed*, which
   resolves to the mediator's own count of that location
   (`interleaver.py:161-167`, `:306-329`, `occurrence` `:331-334`). The first
   hit of a pinned non-zero step relaxes the mediator (`:478-483`).
   `Iterations.__iter__` pins before each step and restores the previous pin
   on exit (`iterator.py:121-144`).
5. **An operation inside a called function exists only after a drill in the
   current run.** `SourceEnvoy.source` writes a `None` placeholder into
   `interleaver.sourced`, parks on `{path}.fn`, receives the live callee from
   `run_op` and stores the instrumented copy back (`source.py:787-833`,
   `run_op` `:495-503`); `interleaver.sourced` is cleared on every run's entry
   (`interleaver.py:710`). The `.fn` handoff happens *before* the call runs,
   and a `Source.__getattr__` for an unknown name is an `AttributeError`
   listing the available ops (`source.py:1014-1033`).

Everything below is a consequence of these five.

## `EProperty`: availability

`EProperty(eproperty)` (`nnter/components/eproperty.py:33-110`) adds one
argument, `unavailable`: a reason string, or a predicate `f(envoy) -> str |
None` (`:43-50`). `reason(obj)` evaluates it on the *instance* (`:60-62`), so
a checkpoint's config decides (`needs_eager`, `components/attention.py:16-21`,
reads `envoy._module.config._attn_implementation`) and so a hybrid can answer
per block.

- `_check` (`:89-100`) runs before every read and write. A reason raises
  `Unavailable` (a `RuntimeError`, `:19-30`) naming `{obj.path}.{name}` and
  the reason. A predicate that itself raises `AttributeError` is re-raised as
  `RuntimeError("the availability check of ... failed: ...")` (`:92-98`),
  because of fact 2: left alone, a typo in a predicate would surface as "no
  attribute `attention_probabilities`" and hide the predicate's own bug.
  Verified: a predicate reading `config.no_such_flag` raises
  `RuntimeError: the availability check of model.model.layers.0.self_attn.probe failed: 'LlamaConfig' object has no attribute ...`.
- `__set_name__` (`:52-58`) fills `name` and `key` from the class body when
  they are still `None`. A bare marker `attention_probabilities =
  unavailable("...")` (`:113-120`) is never called on a stub, so nnsight's
  `__call__` never ran to set them; without this the marker would have no
  name in the repr and `status()`.
- `layout` and `dims` (`:64-87`) read the return annotation of the stub with
  `typing.get_type_hints(func, include_extras=True)`, which evaluates a
  string annotation (the components use `from __future__ import annotations`)
  in the stub's module globals, where `Float` and `Tensor` are imported, and
  returns a real annotation as it is; a `Float[...] | None` annotation (`LinearAttention.state_input`, which is
  `None` on a fresh prompt) is a `types.UnionType`, so the non-`None` member
  is taken (`:79-80`). `dims` is `layout.dim_str.split()`. Verified:
  `LinearAttention.state_input.dims == ('batch', 'heads', 'key_dim', 'value_dim')`.
- `hasattr(envoy, "value")` and `getattr(envoy, "value", None)` both **raise**
  `Unavailable`, since Python's default only swallows `AttributeError`
  (`tests/test_base.py:49-52`). The reason this is not turned into an
  `AttributeError` is fact 2 again: the reason text would be lost. Use
  `status()`.

## `SourceEProperty`: a value inside a forward

`SourceEProperty` (`:123-246`) is an `EProperty` whose location is an
operation under the module's `.source`.

- **Key.** `key = f"source.{op}.{attribute}"` (`:166-167`), the same string
  a user would write after the envoy (`envoy.source.<op>.<attribute>`). When
  `op` is a function (a branching forward, or Falcon's `by_alibi`, whose
  function is named `<without>|<with_alibi>`), the key shows `<name>`.
- **`_drill`** (`:194-207`) walks `op.split(".source.")`: every component
  but the last is a *call* to drill into, the last is the operation whose
  `.output` / `.inputs` / `.input` descriptor is read. The walk starts at
  `obj.source`, which itself instruments the module's forward once
  (`Envoy.source`, nnsight `envoy.py:645-663`) and is fine outside a trace;
  each `.source` on an op needs a running trace (fact 5). An `AttributeError`
  anywhere in the walk is re-raised as `SourceNotAvailable` naming the value,
  the op and nnsight's list of what is there (`:202-207`), again because of
  fact 2. Verified: a wrong op name reads `model.model.layers.0.self_attn.broken reads operation 'no_such_op_0' under .source, which this run does not have: 'model.model.layers.0.self_attn.source' has no operation ...`.
- **Every read and write drills.** Because `interleaver.sourced` is per run
  (fact 5), the descriptor cannot remember a `SourceEnvoy` from a previous
  trace. `_is_drilled` (`:209-211`) checks `getattr(source, call).path in
  obj.interleaver.sourced` to tell "already drilled this run" (walk the cached
  entry with a plain `.source`) from "first time this run" (`_drill_relaxed`).
- **`_drill_relaxed`** (`:213-230`) is fact 4 meeting fact 5. A drill parks
  on `{path}.fn`, a location the call fires **once per forward**. Inside
  `tracer.iter[7]` the mediator is pinned to occurrence 7, so the `.fn` read
  would wait for the eighth firing of a call that fires once, and the run
  would end with it parked. So the pin is saved, set to `None`, the drill
  made relaxed (it resolves to the next occurrence, the call in flight), and
  the pin restored in `finally` for the value read that follows. This is what
  lets `for t in tracer.iter[2]: mix.state` resolve on a first read pinned
  past 0 (`tests/families/test_qwen3_5_text.py:173-176`).
- **`_pick` / `_put`** (`:172-192`) implement `select`: with
  `attribute="inputs"` an int indexes `args` and a str indexes `kwargs`
  (`attention_queries` is `select=1` of the interface's inputs,
  `components/attention.py:74`; a DeltaNet `decays` is `select="g"`,
  `linear_attention.py:176`); with `"output"` an int indexes the returned
  tuple (`attention_head_outputs` is `select=0`, `attention.py:149`). A write
  re-reads the current `(args, kwargs)` or tuple, replaces the one element
  and writes the whole back (`__set__`, `:239-246`), which is why assigning
  `attention_keys` replaces only the keys and keeps every other argument.
- **`__get__` / `__set__`** (`:232-246`) do not call `eproperty.__get__` at
  all: they read `getattr(op, self.attribute)`, i.e. the `SourceEnvoy`'s own
  `.output` / `.inputs` / `.input` eproperties (nnsight `source.py:835-884`),
  and apply nnter's preprocess/postprocess around them. `.input`'s postprocess
  re-reads the pair and replaces the first argument (`source.py:882-884`), so
  BLOOM's `attention_output = value` (a `dropout_add` `.input`) keeps the
  residual argument intact.
- **Why `transform` is not wired here.** Fact 3 is a feature of
  `eproperty.__get__`, which this descriptor bypasses. The element `_pick`
  returns is the very object the call holds, so an in-place edit on it
  reaches the model without a write-back, and a reshaped view would need one
  the descriptor does not provide (`:152-155`). The two families that serve a
  transposed view of head outputs (`seq_first`, `attention.py:35-43`) rely on
  a transpose being a view of the same storage, so in-place edits still land,
  and transpose back in `postprocess` for assignments
  (`families/falcon.py:62-68`, `families/mpt.py:61-67`). A value that needs a
  real write-back stays a plain `EProperty`: Falcon's `mlp_output` reads a
  clone and carries edits back with `@mlp_output.transform`
  (`families/falcon.py:91-103`), and the transform's `raw` is what a
  tuple-returning module would need to rebuild its container
  (nnsight `eproperty.py:52-66`).

## `RelativeEProperty`: a sibling's value

`RelativeEProperty(EProperty)` (`:249-267`) is a plain `eproperty` whose
location is computed from another envoy. `key` is `"<path>.<attribute>"`;
`_location` (`:262-267`) splits on the last dot, then either steps to the
parent by string surgery on `obj.path` when the path starts with `../`
(`"../post_attention_layernorm.output"` on `model.layers.0.self_attn`
becomes `model.layers.0.post_attention_layernorm.output`, by **native**
name since it is a path string), or resolves the path from the envoy with
`obj.get(path).path`, which follows aliases (`"embed_tokens.output"` on the
root reaches `transformer.wte` on GPT-2, `standardized.py:132-140`). Reading
and writing then go through `eproperty.__get__` / `__set__` unchanged, so
`transform` *is* available on this descriptor.

## `DerivedEProperty`: computed, read-only

`DerivedEProperty(EProperty)` (`:315-338`) has no location: `__get__`
(`:331-335`) runs `_check` then `compute(obj)`, which may read any number of
served values in forward order; `__set__` raises `AttributeError("... is
derived and read-only")` (`:337-338`), which is the one place an
`AttributeError` is right (it is the assignment that fails). `_preprocess`
is set to `compute` (`:326`) **only** so that `layout` reads its return
annotation; it is never called as a preprocess. `LinearAttention.states`
(`linear_attention.py:281-285`) is the one instance.

## `branched`, `per_call`, `at_occurrence`

A forward that branches fires a different op on each branch, and the value
of the branch variable is itself served once per module call. Three helpers
(`:269-344`) let a `SourceEProperty` name the op that fires *on this call*:

- `branched(variable, ops)` (`:269-292`) returns an `op` function for
  `SourceEProperty`: read `getattr(envoy.source, variable).output` (a
  binding is an operation, nnsight `source.py:26-31`) and look it up in
  `ops`. The read is cached through `per_call`.
- `per_call(envoy, key, compute)` (`:295-312`) caches `compute()` on the
  envoy under `key`, telling calls apart by fact 4: the cache entry is
  `(mediator, step, value)`; another mediator is another run; a step that is
  `None` (relaxed) is the same call; a pinned step different from the cached
  one is a new call. So inside `tracer.iter[:]` the first read of a step body
  is pinned to that step and misses the cache, and every later read in the
  same body is relaxed and hits it. Reading the variable as the step's first,
  pinned read also keeps the kernel read sequential, which is what lets an op
  that never fires on step 0 resolve on later steps (`:279-285`).
- `at_occurrence(t)` (`:340-344`) is `Iterations()[t : t + 1]`: the
  `for step in tracer.iter[t]` stretch as a value, so library code can pin
  one occurrence of a location and have the pin restored afterwards
  (`iterator.py:121-144`).

[linear-attention-internals.md](linear-attention-internals.md) is where all
three are used.

## Where it lives

| descriptor | class | location it serves | write path |
|---|---|---|---|
| boundary value (`layer_output`, `attention_output`, `logits`) | `EProperty(key="output")` | `{path}.output` (same as `.output`) | `postprocess` → swap; `transform` available |
| value inside a forward (`attention_probabilities`, `decays`) | `SourceEProperty(op, attribute, select)` | `{path}.source.<op>.<attribute>` after a drill | `postprocess` → `_put` → the op's own descriptor |
| a sibling's value (Gemma-2 `attention_output`) | `RelativeEProperty("../norm.output")` | the sibling's `.output` | as `EProperty` |
| computed (`states`) | `DerivedEProperty(compute)` | none | refused |
| declared missing | `unavailable("reason")` | none | refused with the reason |

## Gotchas

- `hasattr` / `getattr(..., default)` raise `Unavailable`; only
  `AttributeError` counts as absence in Python, and turning `Unavailable`
  into one would lose the reason through `Envoy.__getattr__`.
- A nested `.source` (`op.source`) only works inside a trace, and the drill
  parks *before* the call fires; a value read after the op's `.output` cannot
  then drill into it (nnsight `source.py:807-813`, the `.fn` ordering).
- `interleaver.sourced` is cleared on entry to every run, so nothing about a
  drill can be cached on the descriptor across traces; `_drill` runs on every
  access by design.
- An out-of-order read of a source-located value ends the block with
  nnsight's "was never reached … cut short" *warning*, not `OutOfOrderError`,
  because the mediator is relaxed while it waits on `.fn`
  (`dangling_unwind`, nnsight `interleaver.py:554-572`): the rest of the block
  silently does not run. See [gotchas.md](gotchas.md).
- `select` on `"output"` indexes a tuple; on a module that returns a bare
  tensor use no `select`.
- `.source` instruments the forward over a snapshot of the module's globals
  at first drill (nnsight `source.py:447-471`, `function_like` copies
  `fn.__globals__`); a kernel binding switched after that is not seen. See
  `route_delta_rule` in [linear-attention-internals.md](linear-attention-internals.md).

## Related

- [architecture.md](architecture.md) — where the descriptors sit in the tree
- [linear-attention-internals.md](linear-attention-internals.md) — `branched`, `per_call`, `at_occurrence` in use
- [gotchas.md](gotchas.md)
- nnsight `docs/developing/extending-envoy.md`, `docs/developing/source-internals.md`, `docs/developing/interleaver-internals.md`
