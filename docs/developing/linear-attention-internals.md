---
title: Linear Attention Internals
one_liner: How LinearAttention reads a gated DeltaNet mixer — the branch-chosen kernel op, per-token state through occurrence arithmetic, and the process-wide kernel routing .source makes necessary.
tags: [developing, internals, hybrids, deltanet, linear-attention, occurrences]
related: [docs/developing/eproperty-internals.md, docs/developing/architecture.md, docs/developing/gotchas.md, docs/usage/delta-net.md]
sources: [nnter/components/linear_attention.py, nnter/components/eproperty.py, nnter/families/qwen3_5_text.py, tests/families/test_qwen3_5_text.py, nnsight src/nnsight/intervention/interleaver.py, nnsight src/nnsight/intervention/iterator.py]
---

# Linear Attention Internals

## What this is for

`LinearAttention` (`nnter/components/linear_attention.py:112-310`) is the
envoy on a hybrid's gated DeltaNet mixer (`linear_attn` on Qwen3-Next,
Qwen3.5 and Qwen3.5-MoE text). Its values live at a kernel call inside the
mixer's forward, and that forward has three properties the softmax
`Attention` never meets: it calls a **different** kernel on a prompt and on a
decode step, the kernel is a **module global** that transformers rebinds to
an optimized implementation when one is installed, and the state after
every token exists only inside a **loop** in one of the two kernels. This
page is how the component handles each, with the transformers lines it
depends on (`models/qwen3_5/modeling_qwen3_5.py` on the stack
[transformers-compat.md](transformers-compat.md) names; Qwen3-Next is the
same shape).

## Canonical pattern

Occurrences are counted per location over the whole run, so a decode step's
first token is not occurrence 0 once an earlier step has fired the same op.
Run on `yujiepan/qwen3.5-tiny-random`:

```python
import torch
import nnter
from nnter import StandardizedTransformer, route_delta_rule
from nnter.families import qwen3_5_text

route_delta_rule(qwen3_5_text, "recurrent")          # before any trace of a DeltaNet layer
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
mix = model.layers[0].linear_attn
prompt = "Hello world there"
n = len(model.tokenizer(prompt).input_ids)

with model.trace(prompt):
    prompt_states = mix.states.save()                 # [batch, n, heads, key_dim, value_dim]

firsts, seqs, stacks, outs = [], [], [], []           # bound outside: names bound in the block do not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:]:
        seq, first = mix._call()                      # (this call's length, occurrence of its first token)
        seqs.append(seq)
        firsts.append(first)
        stacks.append(mix.states.save())              # through whichever kernel fires this step
        outs.append(mix.state_output.save())

seqs    == [n, 1, 1]                                  # the prompt, then one token per step
firsts  == [0, 0, 1]                                  # the prompt's op starts at 0; step 1 at 0 of the *recurrent* op; step 2 at 1
torch.equal(stacks[0], prompt_states)                 # True
torch.equal(stacks[2][:, 0], outs[2])                 # True: step 2's `states` is step 2's token ...
torch.equal(stacks[2][:, 0], outs[1])                 # False: ... not occurrence 0 of the op, which was step 1's
route_delta_rule(qwen3_5_text, "chunked")
```

The open `tracer.iter[:]` ends with nnsight's "never reached" warning by
design (`nnsight iterator.py:66-71`). Without the `first` offset, step 2
would ask for occurrence 0 of the recurrent kernel's state op, which step 1
already consumed, and the read would dangle.

## Two kernels, one branch variable

The mixer's forward binds `use_precomputed_states` before it branches
(`modeling_qwen3_5.py:561-563`), then calls
`torch_recurrent_gated_delta_rule(...)` when it has a cached state and one
token (`:625-637`) and `torch_chunk_gated_delta_rule(...)` otherwise
(`:638-650`). Both take `(query, key, value, g=, beta=, initial_state=, ...)`
and return `(core_attn_out, last_recurrent_state)`. nnsight names them
`torch_recurrent_gated_delta_rule_0` and `torch_chunk_gated_delta_rule_0`,
and the binding `use_precomputed_states_0` (a binding is an op,
nnsight `source.py:26-31`).

- `CHUNK_KERNEL` / `RECURRENT_KERNEL` (`linear_attention.py:144-146`) are
  those two names.
- `KERNEL = branched("use_precomputed_states_0", {False: CHUNK_KERNEL, True:
  RECURRENT_KERNEL})` (`:148`) is the `op` every kernel-located value is
  declared with (`attention_queries` … `state_output`, `:161-205`). At read
  time it reads the binding's `.output` on this call and picks the name
  ([eproperty-internals.md](eproperty-internals.md), `branched`). The choice
  is cached per call by `per_call`, so the eight values read in one step ask
  the model for the binding once.
- `KERNEL` is a plain function stored on the class, so through an instance
  it is a bound method: `self.KERNEL(self)`, the spelling the declarations
  suggest, passes the envoy twice and raises. Every call site spells it
  `type(self).KERNEL(self)` (`:219`, `:252`, `:264`), the function applied to
  the envoy, which is what the `SourceEProperty` declarations do with it (they
  receive the function itself, before it is a class attribute).

Each decode step under `generate` is one forward over one token: the
kernel's inputs then have sequence length 1, `state_input` is the cached
state, and `state_output` the state the step leaves
(`tests/families/test_qwen3_5_text.py:93-109`).

## `state_input` is a clone

The forward passes the cache's own buffer as `initial_state`
(`modeling_qwen3_5.py:624`, `cache_params.layers[i].recurrent_states[0]`)
and afterwards `cache_params.update_recurrent_state(last_recurrent_state,
...)` (`:653-654`) writes the new state into that buffer in place
(`transformers/cache_utils.py:1091-1093`, "Update the linear attention
cache in-place"). A saved `state_input` that held the live tensor would read
as this step's *output* by the time the trace ends, so the preprocess
returns `value.clone()` (`linear_attention.py:186-195`). Assigning replaces
what the step starts from; the clone only affects reads.

## Optimized kernels have no source

transformers decorates both kernels with
`use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")` /
`("fused_recurrent_gated_delta_rule", "fla")` (`modeling_qwen3_5.py:300`,
`:437`). The decorator (`transformers/integrations/hub_kernels.py:829-870`)
binds the module-level name to a `wrapped` closure whose nonlocals are
`torch_function` (the pure-torch body) and `implementation` (the
`flash-linear-attention` function when that package is installed, else
`torch_function` again).

- `needs_torch_kernels` (`linear_attention.py:79-93`) is the `unavailable`
  predicate of every kernel-located value: it reads each bound name's
  closure with `inspect.getclosurevars(bound).nonlocals` and refuses when
  `implementation is not torch_function`. A compiled kernel has no Python
  source, so `.source` could not drill into it (nnsight `source.py:364-374`,
  `compiled` raises `SourceNotAvailable` for a callable without `__code__`).
  The status text says to uninstall the package.
- `_delta_rule_loop` (`:23-35`) is the same inspection, here to *find* the
  pure-torch token loop: the `torch_function` in the closure of the name
  `RECURRENT_KERNEL` refers to, or the name's own binding when it is already
  a plain function.

## The state after every token: `route_delta_rule`

Only the recurrent kernel has a per-token state. Its loop
(`modeling_qwen3_5.py:480-491`) rebinds `last_recurrent_state` twice per
token, decay at `:484` and update at `:489`; with the two bindings before
the loop (`:474`, `:476`) the post-update binding is the **fourth**
`last_recurrent_state` in the function, so `STATE_OP =
"last_recurrent_state_3"` (`linear_attention.py:150`) fires once per token.
The chunk kernel binds the same name four times too (`:407`, `:409`, `:425`
inside the chunk loop, `:428`), so `last_recurrent_state_3` exists there as
well, but it is the single final binding, not a per-token one; that is why
`state` and `states` are guarded by `needs_recurrent_routing` (`:96-110`)
rather than by the op resolving.

`route_delta_rule(family, kernel)` (`:51-76`) is how a prompt runs through
the loop:

- `_mixer_module(family)` (`:38-48`) finds the transformers modeling module
  from the family's `ENVOYS` entry whose envoy subclasses `LinearAttention`
  (or takes a modeling module directly).
- The module's original bindings of both kernel names are stashed once in
  `module.__dict__["_nnter_delta_rules"]` (`:65-67`), so `"chunked"` can
  restore them (`:72-74`) and repeated `"recurrent"` calls are idempotent.
- `"recurrent"` binds **both** names to the loop `_delta_rule_loop` returns
  (`:68-71`). Both, because the forward's op names stay
  `torch_chunk_gated_delta_rule_0` / `torch_recurrent_gated_delta_rule_0`
  whatever the globals hold: the name in the source is the label, the global
  is what runs. After routing, a prompt's `torch_chunk_gated_delta_rule_0`
  *is* the token loop, and `STATE_OP` under it has one occurrence per token.
- It is process-wide, like installing a kernel: every model of that family in
  the process is affected, and `needs_recurrent_routing` checks the live
  binding (`:103`), so `status()` follows it.

**Why it must run before tracing that layer.** nnsight's `.source` builds
the instrumented forward once per module and, when it drills into a call,
instruments the callee it received and stores it in `interleaver.sourced`
for the run; `instrument` rebuilds the function with `fn.__globals__`
copied into a new function object (nnsight `source.py:439-444`,
`function_like`). A callee already instrumented keeps the binding it was
compiled with, and the module-level `forward` body has already been
compiled over its globals dict, so a rebinding made after the first drill is
not what the instrumented copy calls. The status message and the docstring
say so (`:59-61`); the tests route, load, trace, and restore in a `finally`
(`test_qwen3_5_text.py:118-148`).

## Occurrence arithmetic

`state` is a `SourceEProperty` at `{kernel}.source.last_recurrent_state_3`
(`:222-235`), a location with one occurrence per token, so nnsight's own
`tracer.iter` walks it: `for t in tracer.iter[:n]: mix.state` reads the
state after every prompt token, `tracer.iter[4]` the one after token 4, and
an assignment there is a write the following tokens continue from
(`test_qwen3_5_text.py:150-178`). `states`, `state_after` and
`set_state_after` are the same location addressed by index, and need three
numbers.

- **Which kernel.** `_token_state_op` (`:209-220`) is `state`'s op function.
  Unpinned or pinned to 0 it calls `type(envoy).KERNEL(envoy)` like every
  other value, so a call-level read after a token loop still finds the
  branch decided (cached). Pinned to a *later* token it cannot read the
  branch variable, which fires once per call and whose occurrence 0 is
  already past, so it takes `CHUNK_KERNEL`: the prompt's kernel, the only
  one a token loop walks.
- **Where this call starts.** `_call` (`:237-256`) computes, once per call
  through `per_call`, `(seq, first)`: `seq` is `attention_queries.shape[1]`,
  and `first` is `Mediator.current(location).occurrence(location)` for the
  state op's `.output` location. The trick is *when* it is read. Reading the
  queries parks the worker at the kernel call's start (nnsight
  `interleaver.py:331-334`, `occurrence` is the interleaver's count minus the
  count when this worker started), which is the one moment the state op's
  count is exactly the number of tokens **earlier calls** put through it.
  Occurrences are per location for the whole run (`interleaver.py:629-633`,
  `:785-791`), so on step *k* ≥ 1 of a `generate` the recurrent op's count is
  *k* − 1: the canonical pattern's `[0, 0, 1]`. The prompt's tokens are under
  the chunk kernel's op, a different location, so they do not offset a
  decode step.
- **Reading position `t`.** `_states` (`:267-276`) loops `for t in
  range(seq): for _ in at_occurrence(first + t): states.append(op.output)`
  and stacks on axis 1; `state_after(t)` (`:292-298`) reads one; `set_state_after`
  (`:300-310`) writes one. `at_occurrence(i)` is `Iterations()[i:i+1]`
  (`eproperty.py:340-344`): it pins the mediator to occurrence `i` for the
  body and restores the previous pin after (nnsight `iterator.py:121-144`).
  The first hit relaxes the mediator (`interleaver.py:478-483`), which is why
  `_call` must have parked at the call's start before the loop begins;
  `_token_op` (`:261-265`) reuses `_call`'s numbers so a second per-token
  read in the same call does not re-ask.
- `states` is a `DerivedEProperty` (`:281-285`): read-only, a stack of the
  per-occurrence reads. `_require_state` (`:287-290`) gives `state_after` and
  `set_state_after` the same `Unavailable` a read of `state` would raise.

Reads follow the forward: in one trace, positions before a write come
before it and positions after it come after; `states` reads every position,
so it goes in its own trace
(`test_qwen3_5_text.py:133-144`).

## Verification

`tests/families/test_qwen3_5_text.py` is the executable version of this
page. `HF_HUB_OFFLINE=1 pytest tests/families/test_qwen3_5_text.py -q`
passes 46 tests in about 13 s (17.5 s wall) on CPU. The methods that pin the
claims above:

| claim | test |
|---|---|
| the values follow the branch and the state hands off between steps | `test_values_follow_the_step_under_generate` (`:93-109`) |
| `state_output` is the state the last token leaves | `test_state_output_is_the_state_the_last_token_leaves` (`:79-86`) |
| `states` needs routing, and `status()` says so per block | `test_per_token_state_needs_the_recurrent_kernel` (`:111-116`) |
| `states[:, t] == state_after(t)`; a write flows into later tokens only | `test_per_token_state_with_the_recurrent_kernel` (`:118-148`) |
| `state` walks with the user's own `tracer.iter`; a first read pinned past 0 resolves | `test_state_iterates_with_the_users_own_iter` (`:150-178`) |
| under `generate`, the prompt is an inner loop on step 0 and each later step one token; a decode step's `states` is its own token | `test_per_token_state_within_a_generate` (`:180-205`) |

The canonical pattern above is the offset experiment in isolation; it ran
on `yujiepan/qwen3.5-tiny-random` with the printed `[0, 0, 1]`.

## Gotchas

- Call `route_delta_rule(model.family, "recurrent")` **before** the first
  trace that touches a DeltaNet layer of that family; a forward `.source` has
  already instrumented keeps the binding it was compiled with.
- Restore with `"chunked"` in a `finally` in tests: the routing is
  process-wide and the next test file's prompts would run the slow loop.
- `type(self).KERNEL(self)`, never `self.KERNEL()`: a function stored on a
  class becomes a bound method.
- `state_input` is a clone; edits to the read tensor do not reach the model,
  assignment does.
- A decode step's `states` offsets by earlier *decode* steps
  (`[0, 0, 1, 2, ...]`), not by the prompt: two locations, two counters.
- `states` reads every position: in a trace with a `set_state_after`, read
  it before the write.
- With `flash-linear-attention` or `causal-conv1d` installed every kernel
  value is unavailable; `status()` says so.

## Related

- [eproperty-internals.md](eproperty-internals.md) — `branched`, `per_call`, `at_occurrence`, `_drill_relaxed`
- [gotchas.md](gotchas.md)
- [testing.md](testing.md) — the hybrid test files
- nnsight `docs/usage/iter-all-next.md` — `tracer.iter` semantics; `docs/developing/interleaver-internals.md` — occurrences
