---
title: Mamba-2 State-Space Mixers
one_liner: The `linear_attn` values on Mamba-2 (SSD) blocks — Mamba-2, Nemotron-H, Bamba, Falcon-H1 — what C, B, x and dt are called, the two kernels a prompt and a decode step run, and the state handed between steps.
tags: [usage, hybrid, state-space, mamba2, ssd, state, nemotron_h, bamba, falcon_h1]
related: [docs/usage/delta-net.md, docs/usage/vocabulary.md, docs/usage/availability.md, docs/usage/layouts.md, docs/usage/generation.md, docs/developing/recurrent-mixer-internals.md]
sources: [nnter/components/state_space.py, nnter/components/recurrent.py, nnter/components/layer.py, nnter/families/mamba2.py, nnter/families/nemotron_h.py, nnter/families/bamba.py, nnter/families/falcon_h1.py, tests/families/ssd.py, tests/families/test_mamba2.py, tests/families/test_nemotron_h.py]
---

# Mamba-2 State-Space Mixers

## What this is for

Mamba-2, Nemotron-H (Nemotron-3 Nano and Super), Bamba and Falcon-H1 mix the
sequence in some or all blocks with a Mamba-2 (SSD) mixer, at
`layers[i].linear_attn`, the recurrent mixer's standard name on every hybrid.
SSD is linear attention with one scalar decay per head. Per head, with the
state `h` a `head_dim` by `state_dim` matrix, each token does

```
h = exp(dt * A) * h + dt * x B^T        # decay the state, write the token's value x at key B
y = h C + D * x                         # read it with query C, plus a skip
```

so `nnter.components.StateSpace` gives the mixer the names a gated DeltaNet
(`LinearAttention`, [delta-net.md](delta-net.md)) has: `C` is
`attention_queries`, `B` is `attention_keys`, `x` is `attention_values`, `dt`
is `betas` (the write strength), `A * dt` is `decays` (the log decay), `y` is
`attention_head_outputs`, and the state entering and leaving the layer is
`state_input` / `state_output`. It is a `RecurrentMixer`, so the kernel switch
and the routing below are the base's.

## Canonical pattern

```python
import nnter
from nnter import StandardizedTransformer, route_kernels

route_kernels(nnter.families.nemotron_h, "torch")     # when mamba_ssm is installed: before the first trace
model = StandardizedTransformer("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", attn_implementation="eager")

ssm = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
mix = model.layers[ssm[0]].linear_attn

with model.trace("The Eiffel Tower is in the city of"):
    C = mix.attention_queries.save()        # [batch, seq, groups, state_dim]
    B = mix.attention_keys.save()           # [batch, seq, groups, state_dim]
    x = mix.attention_values.save()         # [batch, seq, heads, head_dim]
    dt = mix.betas.save()                   # [batch, seq, heads], > 0
    log_decay = mix.decays.save()           # [batch, seq, heads], float32, <= 0
    y = mix.attention_head_outputs.save()   # [batch, seq, heads, head_dim]
    state = mix.state_output.save()         # [batch, heads, state_dim, head_dim]: after the last token
    out = mix.attention_output.save()       # [batch, seq, hidden]: what the mixer adds to the stream
```

Decide which blocks hold a Mamba-2 mixer *outside* the trace, as above.

## Which blocks, and what `status()` says

| family | blocks | `linear_attn` is | the rest of the block |
| --- | --- | --- | --- |
| `mamba2` | every block | `mixer` | `norm` as `input_layernorm`; no `self_attn`, no `mlp` |
| `nemotron_h` | `layers_block_type == "linear_attention"` | `mixer` | one sublayer per block: the `mixer` is `linear_attn`, `self_attn` or `mlp` by its class |
| `bamba` | all but `attn_layer_indices` | `mamba` | `self_attn` on the others; `feed_forward` as `mlp` on every block |
| `falcon_h1` | every block, beside `self_attn` | `mamba` | both mixers in parallel, then `feed_forward` as `mlp` |

On Mamba-2 the block is the mixer alone, so `status()` lists no `self_attn.*`
and no `mlp.*` value, and `layers[i].input + linear_attn.attention_output ==
layer_output`. On Nemotron-H each block has one of `linear_attn`, `self_attn`
and `mlp`, and `status()` reports the other two `no ... module on this block`
per block; the identity is the input plus that one sublayer's contribution.
On Falcon-H1 it has four terms, both mixers and the MLP, each mixer's
contribution scaled by its µP multiplier ([families.md](../reference/families.md)).

`state`, `states`, `state_after` and `set_state_after` are unavailable on every
Mamba-2 mixer: `this mixer's kernels do not materialize the state per token`.

## The values

| value | SSD | what it is | layout |
| --- | --- | --- | --- |
| `attention_output` | | what the mixer adds to the residual stream | `Residual`: `batch seq hidden` |
| `attention_queries` | `C` | what reads the state, after the conv and the activation; one per group of heads | `SSDQueries`: `batch seq groups state_dim` |
| `attention_keys` | `B` | where each token writes into the state; one per group of heads | `SSDKeys`: `batch seq groups state_dim` |
| `attention_values` | `x` | what each token writes | `SSDValues`: `batch seq heads head_dim` |
| `betas` | `dt` | `softplus(dt + dt_bias)` (clamped to `time_step_limit` on a prompt): the write strength; derived, read-only | `Gates`: `batch seq heads` |
| `decays` | `A * dt` | the log of how much of the state each token keeps; float32, non-positive; derived, read-only | `Gates`: `batch seq heads` |
| `state_input` | `h` in | the state the call starts from: `None` on a fresh prompt, the cached state on a decode step (a copy) | `State`: `batch heads key_dim value_dim` |
| `state_output` | `h` out | the state after the call's last token: what the next decode step starts from | `State`: `batch heads key_dim value_dim` |
| `attention_head_outputs` | `y` | each head's read of the state plus the `D` skip, before the gated norm and `out_proj` | `SSDHeadOutputs`: `batch seq heads head_dim` |

`groups` is the mixer's `n_groups` (the heads in a group share `B` and `C`),
`heads` its `num_heads`, `head_dim` its `head_dim` and `state_dim` its
`ssm_state_size`. The state is served key side first, `[batch, heads,
state_dim, head_dim]`, the shared `State` layout (`key_dim` is `state_dim`,
`value_dim` is `head_dim`): the transpose of the cache's `[batch, heads,
head_dim, state_dim]`. An assignment is transposed back.

Everything but `attention_output` is read at the scan kernel call. Assign
`attention_queries`, `attention_keys`, `attention_values`,
`attention_head_outputs`, `state_input` or `state_output` to replace them, or
edit the first four in place (`mix.attention_head_outputs[:, -1] = 0` reaches
the model). `betas` and `decays` are computed from the call's `dt`, `dt_bias`
and `A` and are read-only. `state_input` is a clone: the decode kernel updates
the cache's buffer in place.

## Two kernels, one value

A prompt runs `mamba2_chunk_scan`, and each decode step of `generate` runs
`mamba2_selective_state_update`, which takes its arguments in other places and
has no sequence axis. The forward decodes when `use_precomputed_states and
seq_len == 1`; the values read that binding and the call's length once per
call and select from the kernel that fires, and a decode step's tensors are
served with a sequence axis of 1. So the same value works in a `trace` and at
every step of `tracer.iter`, and the state hands off:

```python
entering, leaving = [], []                       # outside the block: a name bound inside does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

entering[0] is None                              # a fresh prompt
torch.equal(leaving[0], entering[1])             # step 1 starts from what step 0 left
```

On a decode step the update is SSD's recurrence over that step's values:
`state_output == exp(decays)[..., None, None] * state_input + betas[..., None,
None] * B ⊗ x`, with `B` repeated from groups to heads
(`tests/families/ssd.py`, `test_state_hands_off_under_generate`). The update
writes the new state into the cache and returns only `y`, so a decode step's
`state_output` is read inside it; assigning it changes both the cache and `y`.

## The kernels have to be the pure-torch ones

transformers dispatches both scans to `mamba_ssm`'s CUDA kernels when that
package is installed; they have no Python source, and every value but
`attention_output` reports `read inside transformers' pure-torch
mamba2_chunk_scan, but this process dispatches it to an optimized kernel
(mamba_ssm) with no Python source; uninstall it, or call
nnter.route_kernels(model.family, 'torch'), to read these`. `route_kernels(family,
"torch")` binds the family's two kernel names to transformers' pure-torch
functions, process-wide; a prompt keeps the chunked scan. Call it before the
first trace of the layer; `route_kernels(family, "default")` restores the
optimized kernels. On a CPU the optimized kernels do not run at all, so a
model with `mamba_ssm` installed needs the routing to run on CPU. Only the
two scan kernels are routed: the values are read at the scan call, so the
short convolution's kernel (`causal_conv1d`, when installed) does not affect
them.

## Gotchas

- **Route before the layer is traced**, with the family module before loading
  (`nnter.families.mamba2`) or `model.family` after; see
  [recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md).
- **`betas` and `decays` are read-only.** Scale the state's write through
  `attention_keys` or `attention_values`, or replace `state_input`.
- **`state_input` is `None` on a fresh prompt.** Save it only when it is not.
- **A prompt run with `use_cache=False` returns no final state**: `state_output`
  raises `Unavailable` there.
- **Falcon-H1 with `mamba_rms_norm` off** passes the gate into the decode
  kernel, so a decode step's `attention_head_outputs` is gated by `silu(z)`
  where a prompt's is not.

## Related

- [delta-net.md](delta-net.md), the gated DeltaNet mixer with the same names, and a per-token state.
- [vocabulary.md](vocabulary.md), where `linear_attn` sits in the standard names.
- [availability.md](availability.md), per-block `status()` on a hybrid.
- [layouts.md](layouts.md), the layouts beside the softmax ones.
- [generation.md](generation.md), values per decode step.
- [recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md), how `StateSpace` sits on `RecurrentMixer`.
