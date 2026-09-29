---
title: Attention Interior
one_liner: Read, edit and assign the queries, keys, values, scores, pattern and per-head outputs inside every family's attention, under `attn_implementation="eager"`.
tags: [usage, attention, interior, source, eager, heads]
related: [docs/usage/residual-stream.md, docs/usage/layouts.md, docs/usage/availability.md, docs/usage/loading.md, docs/usage/generation.md, docs/usage/delta-net.md, docs/usage/remote.md]
sources: [nnter/components/attention.py, nnter/components/eproperty.py, nnter/families/gpt2.py, nnter/families/falcon.py, nnter/families/gpt_oss.py, nnter/families/gptj.py, nnter/families/bloom.py, nnter/families/mpt.py, nnter/families/deepseek_v2.py, tests/families/suite.py]
---

# Attention Interior

## What this is for

`attention_output` is what the attention adds to the residual stream. The six
values on `model.layers[i].self_attn` below are what happens *inside* it: the
queries, keys and values the softmax attention receives, the scores entering
the softmax, the pattern the values are mixed with, and each head's output
before the heads are concatenated and projected. Head ablation, pattern
surgery, key-side steering and query/key patching are all reads and writes of
these. They are source-located values: nnter reaches them through nnsight
`.source` (nnsight docs/usage/source.md) at transformers' shared eager
attention call, or at the family's own operations where it does its own
arithmetic, and presents them with one layout on every family.

## Canonical pattern

```python
from nnter import StandardizedTransformer

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", attn_implementation="eager")
attn = model.layers[10].self_attn

with model.trace("The Eiffel Tower is in the city of"):
    q = attn.attention_queries.save()          # [batch, heads, seq, head_dim]      after RoPE
    k = attn.attention_keys.save()             # [batch, kv_heads, seq, head_dim]   before repeat_kv
    v = attn.attention_values.save()           # [batch, kv_heads, seq, head_dim]
    scores = attn.attention_scores.save()      # [batch, heads, query, key]         masked and scaled
    pattern = attn.attention_probabilities.save()   # [batch, heads, query, key]    what the values are mixed with
    heads = attn.attention_head_outputs.save() # [batch, seq, heads, head_dim]      before concat and o_proj
    attn.attention_head_outputs[:, :, 7] = 0   # ablate head 7 of block 10
    logits = model.logits.save()
```

The six reads are in forward order, so they fit in one trace; a write follows
the same rule (write the head outputs after reading the scores, not before).
`softmax(scores)` equals `pattern` to the dtype cast, and the pattern's rows
sum to one and are lower-triangular.

## The six values

| value | what it is | layout |
| --- | --- | --- |
| `attention_queries` | the queries entering the attention interface, after the query projection and, where the family has one, after the rotary embedding | `Queries`: `batch heads seq qk_head_dim` |
| `attention_keys` | the keys entering the interface, after RoPE and before `repeat_kv`, so the head axis is `num_kv_heads` wide under grouped-query attention | `Keys`: `batch kv_heads seq qk_head_dim` |
| `attention_values` | the values entering the interface, before `repeat_kv` | `Values`: `batch kv_heads seq head_dim` |
| `attention_scores` | the scaled, masked scores that enter the softmax | `Pattern`: `batch heads query key` |
| `attention_probabilities` | the pattern the values are mixed with: the dropout's output after the softmax, in the model dtype, an attention sink's column already dropped | `Pattern`: `batch heads query key` |
| `attention_head_outputs` | what the interface returns, each head's mix of the values, before the reshape to `[batch, seq, hidden]` and the output projection | `HeadOutputs`: `batch seq heads head_dim` |

Sizes come off the root: `model.num_heads`, `model.num_kv_heads`,
`model.head_dim` and `model.qk_head_dim` (`head_dim` except under latent
attention). Every value's layout is on the descriptor, one of the named
aliases in `nnter.components`: `Attention.attention_keys.layout is Keys`, and
`Attention.attention_keys.dims` is `("batch", "kv_heads", "seq", "qk_head_dim")`.
The sequence axis is 2 on the queries, keys and values, the layout transformers
hands its interface, and 1 on the head outputs.

The pattern is read after the dropout, not at the softmax, because that is the
tensor the values are multiplied with on every family: cast back to the model
dtype, and on a sink model without the sink column. Under a sink the rows sum
to less than one (GPT-OSS below).

## Requirement: eager attention

All six live inside the eager attention forward. A model loaded with `sdpa`
or `flash_attention_2` never runs it, and each value says so:

```python
model = StandardizedTransformer("openai-community/gpt2", attn_implementation="sdpa")
model.status(layer=0)["self_attn.attention_probabilities"]
# "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"
```

Reading one raises `nnter.Unavailable` with the same reason, before the model
runs. `attention_output` does not depend on the implementation and stays
available. A load with no `attn_implementation` gets transformers' default,
`sdpa` on every family that supports it, so pass `attn_implementation="eager"`
at load for any of the six ([loading.md](loading.md)). GPT-J, BLOOM and MPT
have no other implementation in transformers and load eager with no flag.

## Reading, editing in place, assigning

Each value is the tensor the forward holds, so an in-place edit reaches the
model without any further step, and an assignment replaces it:

```python
with model.trace(prompt):
    attn.attention_scores[:, :, :, 0] = float("-inf")      # no query may attend to key 0 ...
    pattern = attn.attention_probabilities.save()          # ... so the pattern's first column is zero
    attn.attention_probabilities = my_pattern              # or replace the pattern outright
    logits = model.logits.save()
```

Assigning one argument of the interface replaces just that argument. Setting
`attention_keys` leaves the queries and values as they were:

```python
with model.trace(prompt):
    attn.attention_keys = attn.attention_keys * 0          # only the keys change
    v = attn.attention_values.save()                       # untouched
```

A written pattern must be `[batch, heads, query, key]` in the model dtype; a
written argument must match the shape the interface expects for it. The
model, not nnter, reports a mismatch, from inside the forward.

The per-family suite checks every one of these on every pinned family: the
six shapes, `softmax(scores) == pattern`, that zeroing any interior value
moves the logits, and that an in-place edit of the scores and of the head
outputs lands (`tests/families/suite.py`).

## Family caveats

### GPT-2 and MPT: queries, keys and values are split views

GPT-2's `c_attn` produces one tensor that `split` divides into queries, keys
and values, and MPT's `Wqkv` one that `chunk` divides. torch refuses to edit
such a view in place:

```python
with model.trace(prompt):
    attn.attention_queries[:, 0] = 0
# RuntimeError: Output 0 of Select is a view and is being modified inplace. This view is the output of a
# function that returns multiple views. ...
```

Assign instead: `attn.attention_queries = attn.attention_queries * 0` or any
tensor of the same shape. The scores, the pattern and the head outputs accept
in-place edits on both (GPT-2's head outputs are a transposed view, which
torch allows).

A GPT-2 checkpoint with `reorder_and_upcast_attn` set takes GPT-2's own upcast
attention path, where the shared interface never runs. All six values then
report `this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own
upcast attention path`; `attention_output` stays available.

### Grouped-query attention: two head widths

On a checkpoint with `num_key_value_heads < num_attention_heads` (Llama-3,
Mistral, Qwen), the keys and values are read *before* `repeat_kv`, so their
head axis is `model.num_kv_heads` wide while the queries, the scores, the
pattern and the head outputs are `model.num_heads` wide. A head-wise edit of
the keys touches the group of query heads that shares that key head.

### Falcon: two attention branches, by `config.alibi`

Falcon's attention does its own arithmetic, and `config.alibi` picks one of
two branches with different operations. Each of the six values names its
operation by that flag, and all six need only an eager load on either branch.
Without alibi the queries and keys are the two returns of the rotary
embedding and the values are bound *before* it, so in one trace the values
must be read before the queries or keys:

```python
with model.trace(prompt):
    v = attn.attention_values.save()       # binds first
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
```

The other order raises `OutOfOrderError` naming `value_layer_0`. With alibi
there is no rotary: the three are the `query_layer`, `key_layer` and
`value_layer` bindings, in that order, so read queries, then keys, then
values. The pattern is the softmax itself without alibi (no dropout follows)
and the dropout after the second softmax with it. The head outputs are the
`scores @ values` product on both branches, served `[batch, seq, heads,
head_dim]` as a view: heads-first without alibi, flattened over batch and
heads with it, where the view unflattens them and a write reshapes back. The
7B layout is multi-query: keys and values have one head.

### Falcon-40B layout: keys and values already expanded

The `new_decoder_architecture` layout broadcasts its key/value heads to every
query head before the rotary embedding, so `attention_keys` and
`attention_values` are `num_heads` wide there, although
`model.num_kv_heads` reports the config's `num_kv_heads`.

### GPT-J, BLOOM, MPT: the same six on their own operations

These three do their own attention arithmetic too, and their families map the
same six values onto it: GPT-J's around its `_attn` call, BLOOM's on its
`_reshape` split and `bmm`, MPT's on its `*_states` bindings and second
`matmul`. The head outputs are heads-first in those forwards and are served as
a sequence-first view, so an in-place edit still lands. All three load eager
by default. BLOOM's and MPT's values do not check the implementation at all;
GPT-J's carry the eager check like the interface families'. BLOOM's queries,
keys and values take in-place edits; MPT's are split views (above).

### GPT-OSS: an attention sink

Each GPT-OSS head carries a learned sink logit that joins the softmax as one
extra key column and is dropped afterwards. So the pattern's rows sum to
*less* than one, and `attention_scores` is read at the masked scores just
before the sink column joins them, the same shape as the pattern. A plain
`softmax(scores)` does not reproduce the pattern there; the sink does:

```python
sinks = attn._module.sinks.to(scores.dtype)                          # [heads]
column = sinks.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
combined = torch.cat([scores, column], dim=-1)
combined = combined - combined.max(dim=-1, keepdim=True).values
pattern_again = combined.softmax(-1)[..., :-1]                        # equals attention_probabilities
```

### DeepSeek: multi-head latent attention

DeepSeek-V2/V3 give queries and keys `qk_head_dim = qk_nope_head_dim +
qk_rope_head_dim` and values `v_head_dim`, so `attention_queries` and
`attention_keys` are `model.qk_head_dim` wide and `attention_values` and
`attention_head_outputs` are `model.head_dim` wide. The interface sees
`num_heads` key/value heads whatever `num_key_value_heads` says: the latent
projection produces keys and values for every head.

## Under `generate`

The values are per forward call. On a decode step the queries, the scores,
the pattern and the head outputs have one query position, and the keys and
values (and the pattern's key axis) are as long as the cache:
`attention_probabilities` is `[batch, heads, 1, prompt + step]`. See
[generation.md](generation.md).

## Gotchas

- **`attn_implementation="eager"` or nothing.** Under `sdpa` every interior
  value is `Unavailable` with the reason above; `status()` says so per block
  before any trace.
- **Reads follow the forward within one trace.** Queries, keys and values,
  then scores, then pattern, then head outputs; on Falcon without alibi the
  values before the queries and keys. An out-of-order read raises `OutOfOrderError`, and
  inside a `tracer.iter` body the error can name a later location than the one
  you misplaced.
- **GPT-2 and MPT queries, keys and values: assign, do not edit in place.**
- **`hasattr(attn, "attention_probabilities")` raises `Unavailable`** when the
  value is unavailable; use `attn.status()` or `model.status(layer=i)`.
- **A sink model's pattern rows sum to less than one**, and its
  `attention_scores` are one step before the softmax's own input.
- **On a hybrid, three blocks in four have no `self_attn`.** Decide which blocks
  have one outside the trace; see [delta-net.md](delta-net.md).
- **Remote runs re-resolve these on the server**, against its transformers; see
  [remote.md](remote.md).

## Related

- [residual-stream.md](residual-stream.md), `attention_output`: what these six add up to.
- [layouts.md](layouts.md), every value's axes, and `value.dims`.
- [availability.md](availability.md), `status()` and `Unavailable`.
- [loading.md](loading.md), `attn_implementation` at load.
- [generation.md](generation.md), the same values per decode step.
- [delta-net.md](delta-net.md), the `linear_attn` counterpart on hybrids: queries, keys, values, gates and a recurrent state, no pattern.
- [remote.md](remote.md), what a source-located value needs from an NDIF server.
- nnsight docs/usage/source.md, the mechanism these values are built on.
