---
title: Layouts
one_liner: "Every standard value has one axis layout on every family, one of fourteen named `jaxtyping` types defined beside the envoy that serves them (`Residual`, `Pattern`, `Keys`, ... from `nnter.components`) you can read (`value.dims`), check (`isinstance(t, value.layout)`) and annotate your own values with."
tags: [usage, layouts, shapes, jaxtyping, dims, heads, kv_heads, Residual, Pattern]
related: [docs/usage/root-values.md, docs/usage/residual-stream.md, docs/usage/availability.md, docs/extending/custom-values.md]
sources: [nnter/components/eproperty.py, nnter/components/layer.py, nnter/components/attention.py, nnter/components/linear_attention.py, nnter/standardized.py, nnter/components/__init__.py]
---

# Layouts

## What this is for

A value's shape is part of what it means. Each standard value is annotated with one of
fourteen named layouts, each defined in the file of the envoy that serves it (`Residual` in `nnter/components/layer.py`; `Queries`, `Keys`, `Values`, `Pattern`, `HeadOutputs` in `nnter/components/attention.py`; `LinearQK`, `LinearV`, `Gates`, `State`, `States` in `nnter/components/linear_attention.py`; `Logits`, `NextTokenProbs`, `Tokens` beside the root values in `nnter/standardized.py`); `nnter.components`
re-exports the eleven envoy-level names, and the root's three come from `nnter.standardized`.
They are `jaxtyping` types such as `Residual = Float[Tensor, "batch seq hidden"]` and
`Pattern = Float[Tensor, "batch heads query key"]`. `value.layout` returns that alias itself
and `value.dims` names its axes. Layouts differ between values, not between families:
`attention_probabilities` is a `Pattern`, `[batch, heads, query, key]`, on GPT-2, Llama and
BLOOM alike, and the per-family suite checks every value's axes against the model's sizes.

## Canonical pattern

```python
import torch
from nnter import Attention, Layer, StandardizedTransformer
from nnter.components import Queries, Residual
from nnter.standardized import Logits

Attention.attention_queries.dims        # ('batch', 'heads', 'seq', 'qk_head_dim')
Attention.attention_queries.layout      # jaxtyping.Float[Tensor, 'batch heads seq qk_head_dim']
Attention.attention_queries.layout is Queries   # True: the alias itself, not a copy of it
Layer.layer_output.layout is Residual   # True
StandardizedTransformer.logits.layout is Logits   # True
Layer.layer_output.dims                 # ('batch', 'seq', 'hidden')
StandardizedTransformer.logits.dims     # ('batch', 'seq', 'vocab')

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True, attn_implementation="eager")

with model.trace("The Eiffel Tower is in"):
    q = model.layers[0].self_attn.attention_queries.save()

isinstance(q, Attention.attention_queries.layout)          # True: rank 4, floating dtype
isinstance(q, Queries)                                     # the same check, by name
isinstance(q[0], Attention.attention_queries.layout)       # False: rank 3
isinstance(q.long(), Attention.attention_queries.layout)   # False: not a float
q.shape                                                    # (1, num_heads, seq, head_dim)
```

Read the layout off the class (`Attention.attention_queries`) or off the instance's type
(`type(model.layers[0].self_attn).attention_queries`); the family's subclass inherits the
annotation unless it redefines the value, and a redefinition is annotated with the same
name (`nnter.families.falcon.Attention.attention_keys.layout is Keys`), so it cannot drift
from the base.

## The layouts

The fourteen names, their axes, and the values that carry each:

| layout | axes | values |
| --- | --- | --- |
| `Residual` | `batch seq hidden` | `layer_output`, `attention_output`, `mlp_output`, `token_embeddings` (and the plain `self_attn.input`, `mlp.input`) |
| `Logits` | `batch seq vocab` | `logits` |
| `NextTokenProbs` | `batch vocab` | `next_token_probs` |
| `Tokens` | `batch seq` (`Int`) | `input_ids`, `attention_mask` |
| `Queries` | `batch heads seq qk_head_dim` | `attention_queries` |
| `Keys` | `batch kv_heads seq qk_head_dim` | `attention_keys` |
| `Values` | `batch kv_heads seq head_dim` | `attention_values` |
| `Pattern` | `batch heads query key` | `attention_scores`, `attention_probabilities` |
| `HeadOutputs` | `batch seq heads head_dim` | `attention_head_outputs` |
| `LinearQK` | `batch seq heads key_dim` | `linear_attn.attention_queries`, `attention_keys` |
| `LinearV` | `batch seq heads value_dim` | `linear_attn.attention_values`, `attention_head_outputs` |
| `Gates` | `batch seq heads` | `linear_attn.decays`, `betas` |
| `State` | `batch heads key_dim value_dim` | `state_input`, `state_output`, `state` |
| `States` | `batch seq heads key_dim value_dim` | `states` |

An axis name means the same thing on every layout: `batch` is axis 0 everywhere, `seq`
the token axis, `heads` the query heads and `kv_heads` the key/value heads, `head_dim`
the width of a head's values and outputs and `qk_head_dim` that of its queries and keys
(the same number outside latent attention), `query` and `key` the two token axes of a
pattern, and on a gated DeltaNet mixer `key_dim` / `value_dim` the state's two sides. The
comments above each alias in its defining file state the same.

`isinstance` checks rank and dtype only; the axis *names* are documentation plus what the
suite asserts against `model.num_heads`, `model.num_kv_heads`, `model.head_dim`,
`model.qk_head_dim`, `model.hidden_size` and `model.vocab_size` ([root-values](root-values.md)).

## Annotating with a name

A value you define is annotated with the name, so its layout is the same object as the
base's and reads back through `.layout` and `.dims` like any standard value:

```python
from nnter import SourceEProperty
from nnter.components import Pattern, interface_reason
from nnter.families import gpt2


class Attention(gpt2.Attention):
    @SourceEProperty("attention_interface_1.source.nn_functional_softmax_0", description="The softmax output before the dropout, [batch, heads, query, key]", unavailable=interface_reason)
    def attention_softmax(self, value) -> Pattern:
        return value


Attention.attention_softmax.layout is Pattern                          # True
Attention.attention_softmax.layout is Attention.attention_probabilities.layout   # True
```

A family that redefines a standard value writes the base's name (`-> Keys`, `-> Residual`),
never an inline string, which is what keeps a redefinition from drifting; a value of your
own takes the name where one fits, and an inline `Float[Tensor, "..."]` with the same axis
names where none does (a per-row entropy, `batch heads query`). See
[custom-values](../extending/custom-values.md) and
[overriding-values](../extending/overriding-values.md).

## Where the sequence axis is

Batch is axis 0 everywhere. The sequence axis is 1 on every value that has one, with one
exception: softmax attention's `attention_queries`, `attention_keys` and
`attention_values`, where it is 2, the `[batch, heads, seq, head_dim]` layout transformers
hands its attention interface. `attention_head_outputs` is back to sequence-first,
`[batch, seq, heads, head_dim]`, on every family (a family whose own arithmetic keeps heads
first serves a transposed view). `attention_scores` and `attention_probabilities` have two
sequence axes, `query` then `key`.

So "the last token" is `value[:, -1]` for a residual-stream value, `value[:, :, -1]` for the
queries, keys and values, and `pattern[:, :, -1, :]` for the last query row of the pattern.

## `kv_heads` under grouped-query attention

Keys and values are read before `repeat_kv`, so their head axis is `num_kv_heads` wide, not
`num_heads`:

```python
model = StandardizedTransformer("Qwen/Qwen3-8B", dispatch=True, attn_implementation="eager")
model.num_heads, model.num_kv_heads, model.head_dim     # e.g. (4, 2, 128) on the tiny checkpoint

with model.trace(prompt):
    k = model.layers[0].self_attn.attention_keys.save()
with model.trace(prompt):
    q = model.layers[0].self_attn.attention_queries.save()
k.shape, q.shape       # (1, 2, seq, 128), (1, 4, seq, 128)
```

Two families expand the key/value heads before the interface, so `kv_heads` reads as
`num_heads` there: Falcon's 40B layout (`num_kv_heads` 8 in the config, 128 heads on the
tensor) and DeepSeek's latent attention.

## `qk_head_dim` on DeepSeek

Multi-head latent attention gives queries and keys a different width from values:
`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`, and `head_dim` is `v_head_dim`. The
root publishes both:

```python
model = StandardizedTransformer("deepseek-ai/DeepSeek-V3", attn_implementation="eager")
model.head_dim, model.qk_head_dim        # (128, 192)

# one value per trace: the interface serves them at one point of the forward
with model.trace(prompt):
    q = model.layers[0].self_attn.attention_queries.save()       # (1, heads, seq, 192)
with model.trace(prompt):
    k = model.layers[0].self_attn.attention_keys.save()          # (1, heads, seq, 192)
with model.trace(prompt):
    v = model.layers[0].self_attn.attention_values.save()        # (1, heads, seq, 128)
with model.trace(prompt):
    h = model.layers[0].self_attn.attention_head_outputs.save()  # (1, seq, heads, 128)
```

The keys share the queries' width, so their last axis is `qk_head_dim` too; the two sizes
coincide on every family without latent attention.

## `heads` on DeltaNet values

On a hybrid's `linear_attn`, `heads` is the mixer's value-head count (`num_v_heads`), which
is what the keys are repeated to before the delta rule; `key_dim` and `value_dim` are the
mixer's `head_k_dim` and `head_v_dim`, not the root's `head_dim`:

```python
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
mix = model.layers[0].linear_attn

with model.trace(prompt):
    q = mix.attention_queries.save()        # (1, seq, num_v_heads, head_k_dim)   bf16
with model.trace(prompt):
    g = mix.decays.save()                   # (1, seq, num_v_heads)               float32
with model.trace(prompt):
    s = mix.state_output.save()             # (1, num_v_heads, head_k_dim, head_v_dim)  float32
```

On the tiny checkpoint the root says `num_heads=8`, `num_kv_heads=4`, and the mixer has
`num_v_heads=8`, `num_k_heads=4`, `head_k_dim=32`, `head_v_dim=32`; the values come out
`(1, 6, 8, 32)`, `(1, 6, 8)` and `(1, 8, 32, 32)` for a 6-token prompt. `state_input` is
`None` on a fresh prompt (no state enters), so `isinstance` on it is meaningless there.
`decays` and the states are float32 whatever the model's dtype; the rest follow the model.

## Gotchas

- **`isinstance(t, value.layout)` checks rank and dtype, not axis sizes.** Compare sizes
  against the root's sizes yourself, or trust the suite.
- **Sequence is axis 2 on `attention_queries` / `attention_keys` / `attention_values`**, axis
  1 everywhere else.
- **`kv_heads` is `num_kv_heads` except where the family expands first** (Falcon 40B layout,
  DeepSeek), where it is `num_heads`.
- **A `.layout` is `None` for a value with no tensor annotation**; `dims` is then `None` too.
- **The names live in `nnter.components`, not `nnter`**: `from nnter.components import Residual`;
  the root's three (`Logits`, `NextTokenProbs`, `Tokens`) only in `nnter.standardized`.
- **Read one interior value per trace when in doubt.** The five interior values bind at
  different points of the forward on the off-interface families (Falcon: values before
  queries and keys).

## Related

- [root-values](root-values.md): the sizes each axis is checked against.
- [residual-stream](residual-stream.md): the `Residual` (`batch seq hidden`) values.
- [availability](availability.md): a value has a layout whether or not this checkpoint has it.
- [custom-values](../extending/custom-values.md): annotating a value of your own with a name.
