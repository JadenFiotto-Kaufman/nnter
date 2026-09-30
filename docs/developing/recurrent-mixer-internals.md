---
title: Recurrent Mixer Internals
one_liner: How RecurrentMixer reaches a recurrent mixer's values — the branch-chosen kernel op, the pure-torch kernels .source needs, process-wide kernel routing, and per-token state through occurrence arithmetic — and how LinearAttention (gated DeltaNet) sits on it.
tags: [developing, internals, hybrids, deltanet, linear-attention, recurrent, occurrences]
related: [docs/developing/eproperty-internals.md, docs/developing/architecture.md, docs/developing/gotchas.md, docs/usage/delta-net.md]
sources: [nnter/components/recurrent.py, nnter/components/linear_attention.py, nnter/components/eproperty.py, nnter/families/qwen3_5_text.py, tests/families/test_qwen3_5_text.py, tests/test_base.py, nnsight src/nnsight/intervention/interleaver.py, nnsight src/nnsight/intervention/iterator.py]
---

# Recurrent Mixer Internals

## What this is for

A recurrent mixer (a gated DeltaNet, a state-space layer) runs its sequence
through a kernel function its forward calls, and its values are the
arguments and results of that call. The forward has three properties the
softmax `Attention` never meets: it calls a **different** kernel on a prompt
and on a decode step, the kernel is a **module global** that transformers
rebinds to an optimized implementation when one is installed, and the state
after every token exists, if at all, only inside a **loop** in one of the
two kernels. How a mixer's values are reached through all three is the same
whatever the values are, so it lives in one base class and the mixers
declare only their values:

- `RecurrentMixer` (`nnter/components/recurrent.py:196-357`) holds the
  mechanism: the kernel choice, `attention_output`, the availability
  predicates, the per-token state machinery and the routing functions.
- `LinearAttention` (`nnter/components/linear_attention.py:23-96`), the
  gated DeltaNet mixer (`linear_attn` on Qwen3-Next, Qwen3.5, Qwen3.5-MoE
  text and OLMo-Hybrid), sets the kernel constants and declares its eight
  values: `attention_queries`, `attention_keys`, `attention_values`,
  `decays`, `betas`, `state_input`, `attention_head_outputs`,
  `state_output`.

This page is how the base handles each property, told through the DeltaNet
subclass with the transformers lines it depends on
(`models/qwen3_5/modeling_qwen3_5.py` on the stack
[transformers-compat.md](transformers-compat.md) names; Qwen3-Next is the
same shape).

## Canonical pattern

Occurrences are counted per location over the whole run, so a decode step's
first token is not occurrence 0 once an earlier step has fired the same op.
Run on `yujiepan/qwen3.5-tiny-random`:

```python
import torch
import nnter
from nnter import StandardizedTransformer, route_kernels
from nnter.families import qwen3_5_text

route_kernels(qwen3_5_text, "torch")                  # before any trace of a DeltaNet layer
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
route_kernels(qwen3_5_text, "default")
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

A subclass names these in class constants (`recurrent.py:227-236`):

| constant | meaning | `LinearAttention` |
|---|---|---|
| `BRANCH` | the binding the forward makes before it branches, `False` on a prompt, `True` on a decode step | `"use_precomputed_states_0"` (the base's default) |
| `CHUNK_KERNEL` | the call a prompt runs through | `"torch_chunk_gated_delta_rule_0"` |
| `RECURRENT_KERNEL` | the call a decode step runs through | `"torch_recurrent_gated_delta_rule_0"` |
| `STATE_OP` | inside the token-by-token kernel, the binding of the state after each token's update, or `None` | `"last_recurrent_state_3"` |

- `KERNEL` is built from them in `__init_subclass__` (`:238-243`) as
  `branched(BRANCH, {False: CHUNK_KERNEL, True: RECURRENT_KERNEL})`, when a
  class sets `BRANCH`, `CHUNK_KERNEL` or `RECURRENT_KERNEL` in its own body
  and not `KERNEL`; a family's subclass of `LinearAttention` inherits it.
  A forward whose branch is not one boolean sets `KERNEL` itself (Mamba-2
  decodes through the recurrent kernel only when `use_precomputed_states
  and seq_len == 1`). It is stored as a `staticmethod`, so
  `type(self).KERNEL(self)` and `self.KERNEL(self)` are the same call.
- `kernel(attribute)` (`:186-193`) is the key function every
  kernel-located value is declared on: it returns
  `source.<KERNEL(envoy)>.<attribute>`, so `LinearAttention` declares its
  values as `@EProperty(kernel("inputs"), select=0, ...)` through
  `@EProperty(kernel("output"), select=1, ...)`
  (`linear_attention.py:51-95`). At read time `KERNEL` reads the binding's
  `.output` on this call and picks the name
  ([eproperty-internals.md](eproperty-internals.md), `branched`). The
  choice is cached per call by `per_call`, so the eight values read in one
  step ask the model for the binding once.
- `attention_output` (`:245-252`) is the base's: the module's own output,
  the contribution to the residual stream, unwrapped from a tuple and
  rewrapped on write.

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
returns `value.clone()` (`linear_attention.py:76-85`). Assigning replaces
what the step starts from; the clone only affects reads. It is a DeltaNet
value, so it is the subclass's; a mixer whose kernel takes the cached state
the same way clones it the same way.

## Optimized kernels have no source

transformers decorates both kernels with
`use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")` /
`("fused_recurrent_gated_delta_rule", "fla")` (`modeling_qwen3_5.py:300`,
`:437`); Mamba-2's `mamba2_chunk_scan` / `mamba2_selective_state_update`
carry the same decorator over `mamba_ssm`. The decorator
(`transformers/integrations/hub_kernels.py:829-870`) binds the
module-level name to a `wrapped` closure whose nonlocals are
`torch_function` (the pure-torch body) and `implementation` (the optimized
function when its package is installed, else `torch_function` again).

- `_dispatch(bound)` (`recurrent.py:43-50`) reads that closure with
  `inspect.getclosurevars(bound).nonlocals`, `{}` for a plain function;
  `_torch_function(bound)` (`:53-55`) is its `torch_function`, or the
  binding itself. `_name(op)` (`:38-40`) is the module-level name an op
  calls (`torch_chunk_gated_delta_rule_0` -> `torch_chunk_gated_delta_rule`).
- `needs_torch_kernels` (`:142-156`) is the `unavailable` predicate of every
  kernel-located value: for `type(envoy).CHUNK_KERNEL` and
  `RECURRENT_KERNEL` it refuses when `implementation is not
  torch_function`. A compiled kernel has no Python source, so `.source`
  could not drill into it (nnsight `source.py:364-374`, `compiled` raises
  `SourceNotAvailable` for a callable without `__code__`). The status text
  names the package and says to uninstall it or route to the torch kernels.

## Routing the kernels: `route_kernels`

`route_kernels(family, kernel)` (`:92-125`) rebinds the family's kernel
names in its modeling module, process-wide:

- `_mixer(family)` (`:62-89`) finds the transformers modeling module and
  the mixer's envoy class from the family's `ENVOYS` entry whose envoy
  subclasses `RecurrentMixer`. Given a modeling module directly, it takes
  the `RecurrentMixer` subclass a loaded family keys on one of that
  module's classes, else one whose `CHUNK_KERNEL` the module defines.
- The module's original bindings of both names are stashed once in
  `module.__dict__["_nnter_kernels"]`, so `"default"` restores them and
  repeated `"torch"` calls are idempotent.
- `"torch"` binds the names to the pure-torch kernels. On a mixer with a
  `STATE_OP`, **both** names are bound to the token-by-token kernel, the
  `torch_function` behind `RECURRENT_KERNEL`: the forward's op names stay
  `torch_chunk_gated_delta_rule_0` / `torch_recurrent_gated_delta_rule_0`
  whatever the globals hold (the name in the source is the label, the
  global is what runs), so after routing a prompt's
  `torch_chunk_gated_delta_rule_0` *is* the token loop and `STATE_OP` under
  it has one occurrence per token. On a mixer without a `STATE_OP`, each
  name is bound to its own `torch_function`: the kernels become readable
  and a prompt keeps its chunked kernel.
- `route_delta_rule(family, kernel)` (`:128-139`) is the DeltaNet spelling:
  `"recurrent"` is `"torch"`, `"chunked"` is `"default"`.

**Why it must run before tracing that layer.** nnsight's `.source` builds
the instrumented forward once per module and, when it drills into a call,
instruments the callee it received and stores it in `interleaver.sourced`
for the run; `instrument` rebuilds the function with `fn.__globals__`
copied into a new function object (nnsight `source.py:439-444`,
`function_like`). A callee already instrumented keeps the binding it was
compiled with, and the module-level `forward` body has already been
compiled over its globals dict, so a rebinding made after the first drill is
not what the instrumented copy calls. The status message and the docstring
say so; the tests route, load, trace, and restore in a `finally`
(`test_qwen3_5_text.py:118-148`), and `tests/test_base.py` checks that
`"torch"` / `"default"` round-trip the bindings.

## The state after every token

Only the recurrent kernel has a per-token state. Its loop
(`modeling_qwen3_5.py:480-491`) rebinds `last_recurrent_state` twice per
token, decay at `:484` and update at `:489`; with the two bindings before
the loop (`:474`, `:476`) the post-update binding is the **fourth**
`last_recurrent_state` in the function, so `STATE_OP =
"last_recurrent_state_3"` (`linear_attention.py:49`) fires once per token.
The chunk kernel binds the same name four times too (`:407`, `:409`, `:425`
inside the chunk loop, `:428`), so `last_recurrent_state_3` exists there as
well, but it is the single final binding, not a per-token one; that is why
`state` and `states` are guarded by `needs_recurrent_routing`
(`recurrent.py:159-176`) rather than by the op resolving. The predicate
answers, in order:

1. `STATE_OP is None`: "this mixer's kernels do not materialize the state
   per token". A subclass without a per-token kernel declares nothing; its
   `state`, `states`, `state_after` and `set_state_after` report that.
2. `needs_torch_kernels`' reason, when an optimized kernel is bound.
3. The live binding of `CHUNK_KERNEL`'s name is not the token-by-token
   loop: the instruction to call `route_kernels(model.family, 'torch')`.
   Checked on the live binding, so `status()` follows the routing.

## Occurrence arithmetic

`state` is an `EProperty` whose key function returns
`source.{kernel}.source.{STATE_OP}.output` (`:256-283`), a location with
one occurrence per token, so nnsight's own `tracer.iter` walks it:
`for t in tracer.iter[:n]: mix.state` reads the state after every prompt
token, `tracer.iter[4]` the one after token 4, and an assignment there is a
write the following tokens continue from
(`test_qwen3_5_text.py:150-178`). `states`, `state_after` and
`set_state_after` are the same location addressed by index, and need three
numbers.

- **Which kernel.** `_token_state_op` (`:256-268`) is `state`'s key
  function. Unpinned or pinned to 0 it calls `KERNEL` like every other
  value, so a call-level read after a token loop still finds the branch
  decided (cached). Pinned to a *later* token it cannot read the branch
  variable, which fires once per call and whose occurrence 0 is already
  past, so it takes `CHUNK_KERNEL`: the prompt's kernel, the only one a
  token loop walks.
- **Where this call starts.** `_call` (`:289-308`) computes, once per call
  through `per_call`, `(seq, first)`: `seq` is `_seq()`, and `first` is
  `Mediator.current(location).occurrence(location)` for the state op's
  `.output` location. `_seq()` (`:285-287`) is an overridable method; the
  base reads `attention_queries.shape[1]`, the DeltaNet kernel's sequence
  axis, and a mixer whose kernel is laid out otherwise overrides it. The
  trick is *when* it is read. Reading a kernel argument parks the worker at
  the kernel call's start (nnsight `interleaver.py:331-334`, `occurrence`
  is the interleaver's count minus the count when this worker started),
  which is the one moment the state op's count is exactly the number of
  tokens **earlier calls** put through it. Occurrences are per location for
  the whole run (`interleaver.py:629-633`, `:785-791`), so on step *k* ≥ 1
  of a `generate` the recurrent op's count is *k* − 1: the canonical
  pattern's `[0, 0, 1]`. The prompt's tokens are under the chunk kernel's
  op, a different location, so they do not offset a decode step.
- **Reading position `t`.** `_states` (`:316-325`) loops `for t in
  range(seq): for _ in at_occurrence(first + t): states.append(op.output)`
  and stacks on axis 1; `state_after(t)` (`:341-347`) reads one;
  `set_state_after` (`:349-357`) writes one. `at_occurrence(i)`
  (`:179-183`) is `Iterations()[i:i+1]`: it pins the mediator to
  occurrence `i` for the body and restores the previous pin after (nnsight
  `iterator.py:121-144`). The first hit relaxes the mediator
  (`interleaver.py:478-483`), which is why `_call` must have parked at the
  call's start before the loop begins; `_token_op` (`:310-314`) reuses
  `_call`'s numbers so a second per-token read in the same call does not
  re-ask.
- `states` is a `DerivedEProperty` (`:330-334`): read-only, a stack of the
  per-occurrence reads. `_require_state` (`:336-339`) gives `state_after`
  and `set_state_after` the same `Unavailable` a read of `state` would
  raise.

Reads follow the forward: in one trace, positions before a write come
before it and positions after it come after; `states` reads every position,
so it goes in its own trace
(`test_qwen3_5_text.py:133-144`).

## Adding a recurrent mixer

A new mixer subclasses `RecurrentMixer`, sets `CHUNK_KERNEL` and
`RECURRENT_KERNEL` (and `BRANCH`, or `KERNEL` itself, when its forward
branches on something else), sets `STATE_OP` only when a kernel updates the
state once per token in a binding, and declares its values at
`kernel("inputs")` / `kernel("output")` with
`unavailable=needs_torch_kernels`. `route_kernels`, the predicates,
`attention_output` and the per-token state come from the base; a kernel
whose tensors are not `[batch, seq, ...]` overrides `_seq`.

## Verification

`tests/families/test_qwen3_5_text.py` is the executable version of this
page, with `tests/test_base.py` for the base on its own.
`HF_HUB_OFFLINE=1 pytest tests/families/test_qwen3_5_text.py -q` passes 46
tests in about 13 s on CPU. The methods that pin the claims above:

| claim | test |
|---|---|
| the values follow the branch and the state hands off between steps | `test_values_follow_the_step_under_generate` (`:93-109`) |
| `state_output` is the state the last token leaves | `test_state_output_is_the_state_the_last_token_leaves` (`:79-86`) |
| `states` needs routing, and `status()` says so per block | `test_per_token_state_needs_the_recurrent_kernel` (`:111-116`) |
| `states[:, t] == state_after(t)`; a write flows into later tokens only | `test_per_token_state_with_the_recurrent_kernel` (`:118-148`) |
| `state` walks with the user's own `tracer.iter`; a first read pinned past 0 resolves | `test_state_iterates_with_the_users_own_iter` (`:150-178`) |
| under `generate`, the prompt is an inner loop on step 0 and each later step one token; a decode step's `states` is its own token | `test_per_token_state_within_a_generate` (`:180-205`) |
| a mixer with no `STATE_OP` reports `state`/`states` unavailable and still serves its kernel values | `tests/test_base.py::test_recurrent_mixer_without_a_state_op_reports_the_state_unavailable` |
| `route_kernels` `"torch"` / `"default"` round-trip the module's bindings, and `route_delta_rule` spells the same | `tests/test_base.py::test_route_kernels_round_trips_the_bindings` |

The canonical pattern above is the offset experiment in isolation; it ran
on `yujiepan/qwen3.5-tiny-random` with the printed `[0, 0, 1]`.

## Gotchas

- Call `route_kernels(model.family, "torch")` **before** the first trace
  that touches a recurrent layer of that family; a forward `.source` has
  already instrumented keeps the binding it was compiled with.
- Restore with `"default"` in a `finally` in tests: the routing is
  process-wide and the next test file's prompts would run the slow loop.
- `state_input` is a clone; edits to the read tensor do not reach the model,
  assignment does.
- A decode step's `states` offsets by earlier *decode* steps
  (`[0, 0, 1, 2, ...]`), not by the prompt: two locations, two counters.
- `states` reads every position: in a trace with a `set_state_after`, read
  it before the write.
- With `flash-linear-attention` or `causal-conv1d` installed every kernel
  value is unavailable; `status()` says so.

## Related

- [eproperty-internals.md](eproperty-internals.md) — `branched`, `per_call`, `_drill` and the relaxed mediator
- [gotchas.md](gotchas.md)
- [testing.md](testing.md) — the hybrid test files
- nnsight `docs/usage/iter-all-next.md` — `tracer.iter` semantics; `docs/developing/interleaver-internals.md` — occurrences
