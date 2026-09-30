---
title: Residual stream and contributions
one_liner: "`layer_output`, `attention_output` and `mlp_output` are tensors on every family, defined by `layers[i].input + attention_output + mlp_output == layer_output`."
tags: [usage, residual-stream, layer_output, attention_output, mlp_output, contributions]
related: [docs/usage/vocabulary.md, docs/usage/methods.md, docs/usage/root-values.md, docs/usage/availability.md]
sources: [nnter/components/layer.py, nnter/components/attention.py, nnter/components/mlp.py, nnter/components/standard.py, nnter/components/eproperty.py, nnter/families/gemma2.py, nnter/families/bloom.py, nnter/families/mpt.py, nnter/families/falcon.py]
---

# Residual stream and contributions

## What this is for

Three standard values give every decoder block the same three tensors:

| value | what it is |
| --- | --- |
| `model.layers[i].layer_output` | the residual stream leaving the block |
| `model.layers[i].self_attn.attention_output` | what the attention sublayer adds to the residual stream |
| `model.layers[i].mlp.mlp_output` | what the MLP sublayer adds to the residual stream |

All three are `[batch, seq, hidden]`, the `Residual` layout (`Layer.layer_output.layout is
nnter.components.Residual`; see [layouts](layouts.md)).

`attention_output` and `mlp_output` are *contributions*, defined by one identity that
holds on a sequential block and a parallel block alike:

```
layers[i].input + attention_output + mlp_output == layers[i].layer_output
```

Raw nnsight `.output` on the same modules does not have that meaning everywhere: a block
may return a tuple, an attention module returns `(attn_output, attn_weights)`, a sandwich
block adds a norm's output rather than the module's, and some modules add the residual
inside. The three values put the same thing at the same name on every family, and the
per-family test suite checks the identity on each.

## Canonical pattern

```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)   # the identity, in the block's dtype
```

Run on the GPT-2 and Gemma-2 tiny checkpoints this gives a maximum absolute difference of
`0.0` in float32 and bfloat16 alike; Falcon's bf16 block sums in another order and lands
within a few bf16 ulps (`torch.testing.assert_close` with `rtol=atol=8 * eps` of the
block's dtype passes, as the suite checks).

Read the four in forward order within one trace: the block's `input`, then the attention,
then the MLP, then the block's output.

## Tensor blocks and tuple blocks

A Llama, GPT-2 or GPT-NeoX block returns `hidden_states` alone. A GPT-J, GPT-Neo, BLOOM, MPT
or Falcon block returns a tuple with it first. `layer_output` is the tensor either way, the
same object the block returned:

```python
model = StandardizedTransformer("bigscience/bloom-560m", dispatch=True)

with model.trace(prompt):
    raw = model.layers[2].output.save()          # a tuple on BLOOM
    out = model.layers[2].layer_output.save()    # the tensor

isinstance(raw, tuple), torch.equal(raw[0], out)   # (True, True)
```

Assigning `layer_output` on a tuple block replaces the first element and leaves the others
as they were, so the next block receives a well-formed tuple:

```python
with model.trace(prompt):
    model.layers[2].layer_output = model.layers[2].layer_output * 0
    after = model.layers[2].output.save()        # (zeros, <the other element, unchanged>)
    nxt = model.layers[3].input.save()           # zeros
```

The same holds for `attention_output` on an attention module that returns
`(attn_output, attn_weights)`: the value is the first tensor, and an assignment rewraps it.

## Reading, editing in place, assigning

All three values go through the interleaver the way `.output` does. In-place edits reach
the model because the value is the live tensor; assignment replaces it:

```python
with model.trace(prompt):
    model.layers[5].self_attn.attention_output[:, -1] = 0     # ablate attention at the last position
    model.layers[5].mlp.mlp_output[:] = 0                     # ablate the MLP everywhere
    model.layers[5].layer_output[:, -1] += direction          # steer the stream leaving the block
    logits = model.logits.save()
```

```python
with model.trace(prompt):
    resid = model.layers[3].layer_output.save()
    model.layers[4].layer_output = resid * 2                  # assign a new tensor
```

`model.layers[i].input` is the residual stream entering the block, a tensor on every
family (the block's first argument is `hidden_states` throughout the registered families).

## Where the families differ

Three shapes of block put the contribution somewhere other than the module's own output,
and the family's `Attention` or `Mlp` subclass points the value at the right place, so the
name means the same thing everywhere. Contrast each with the raw `.output`:

**Sandwich norms (Gemma-2, Gemma-3, OLMo-2, OLMo-3).** The block adds
`post_attention_layernorm(attn(...))` and `post_feedforward_layernorm(mlp(...))`. What
reaches the residual stream is the post-norm's output, so `attention_output` is
`post_attention_layernorm.output` and `mlp_output` is `post_feedforward_layernorm.output`
(an `EProperty` keyed `"../post_attention_layernorm.output"`, the sibling norm):

```python
model = StandardizedTransformer("google/gemma-2-2b", dispatch=True)

with model.trace(prompt):
    raw_attn = model.layers[0].self_attn.output.save()                      # (tensor, weights): before the post-norm
    post = model.layers[0].post_attention_layernorm.output.save()
with model.trace(prompt):
    attn = model.layers[0].self_attn.attention_output.save()

torch.equal(attn, post), torch.equal(attn, raw_attn[0])                     # (True, False)
```

**Residual added inside the module (BLOOM both sublayers, MPT's MLP; DBRX adds outside
the attention module, so its base holds).** The module's output is already a
residual-stream state: BLOOM's `self_attention.output[0]` equals
`layers[i].input + attention_output`. The contribution is the tensor entering the add: the
first argument of `dropout_add` on BLOOM, the MLP dropout's output on MPT (an
`EProperty` keyed inside the forward, `"source.dropout_add_0.input"`):

```python
model = StandardizedTransformer("bigscience/bloom-560m", dispatch=True)

with model.trace(prompt):
    x = model.layers[2].input.save()
    attn = model.layers[2].self_attn.attention_output.save()
with model.trace(prompt):
    raw_attn = model.layers[2].self_attn.output.save()

torch.allclose(raw_attn[0], x + attn)                                        # True: the module returns x + contribution
```

Because the contribution is an operation inside the module, read it before the module's
own `.output` in one trace, or in a trace of its own as above.

**Falcon's copy.** Falcon's block sums `x + attn + mlp` by adding the attention output
*into the MLP's output tensor in place*, so by the time the trace ends the live
`mlp.output` no longer holds the MLP's contribution. `mlp_output` reads a copy taken as the
MLP returns, and an `eproperty` transform carries in-place edits to that copy back into the
model, so both editing forms still land:

```python
model = StandardizedTransformer("tiiuae/falcon-7b", dispatch=True)

with model.trace(prompt):
    model.layers[0].mlp.mlp_output[:] = 0                # reaches the logits
with model.trace(prompt):
    model.layers[0].mlp.mlp_output = model.layers[0].mlp.mlp_output * 0   # also reaches the logits
```

A raw `mlp.output.save()` on Falcon is the live tensor, and it reads as `mlp + attn` after
the block has run; `mlp_output` is the MLP's contribution.

## Gotchas

- **Forward order within one trace.** `layers[i].input`, then `self_attn.attention_output`,
  then `mlp.mlp_output`, then `layer_output`. On BLOOM and MPT a contribution is an
  operation inside the module, so it comes before that module's `.output`; on Gemma-2/3
  and OLMo-2/3 it is the post-norm's output, so it comes after `self_attn.output`. Reading
  the raw and the standard value of one module in one trace forces you to know which; a
  second trace does not. On GPT-Neo `self_attn` is the inner `attn.attention`, so its
  `attention_output` comes before the `attn` wrapper's `.output`.
- **A tuple block's `.output` is a tuple; `layer_output` is the tensor.** Skip a block with
  `Layer.skip_with` ([methods](methods.md)) rather than `.skip(tensor)`, which would hand a
  bare tensor where a tuple is expected.
- **`mlp_output` does not exist on OPT** (no MLP module); `status()` says so
  ([availability](availability.md)).
- **The identity is exact in float32 and within a few ulps in bf16** when the block sums in
  another order (Falcon). Compare with a tolerance in the block's dtype.
- **Nothing bound inside a trace survives without `.save()`**, including the value you read
  to compute a difference; save each operand.

## Related

- [vocabulary](vocabulary.md): `self_attn.input`, `mlp.input`.
- [methods](methods.md): `skip_layers`, `steer` over `layer_output`.
- [root-values](root-values.md): `token_embeddings`, the stream before block 0.
- [availability](availability.md): which blocks have which value.
- nnsight docs/usage/access-and-modify.md: `.output`, in-place edits and assignment.
