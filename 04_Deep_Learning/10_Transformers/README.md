# Transformers

A Transformer drops recurrence entirely. Where [`RNN`](../06_RNN)/[`LSTM`](../07_LSTM)/
[`GRU`](../08_GRU) process a sequence step-by-step, and [`Attention`](../09_Attention)
adds one attention layer on top of recurrence, a Transformer is built **purely** from
self-attention: every token attends to every other token in parallel, using several
attention "heads" at once (multi-head self-attention) so different heads can learn
different kinds of relationships. A standard encoder block is:

```
x -> MultiHeadSelfAttention -> + residual -> LayerNorm
  -> FeedForward             -> + residual -> LayerNorm
```

Since there's no recurrence to track position, a **positional embedding** is added to
the token embedding so the model knows token order.

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `transformers_pytorch.ipynb` | PyTorch | `sklearn.datasets.fetch_20newsgroups` (`rec.sport.hockey` vs `sci.space`) | Binary text classification |
| `transformers_tensorflow.ipynb` | TensorFlow / Keras | same | same |

Built entirely from library components — `nn.TransformerEncoderLayer` (PyTorch, which
wraps `nn.MultiheadAttention`) and `tf.keras.layers.MultiHeadAttention` (TF), no
attention math written by hand. One encoder block on top of token + positional
embeddings, mean-pooled over real (non-padding) tokens, then a sigmoid classifier.

Same `fetch_20newsgroups` pipeline as `06_RNN`–`09_Attention`, with one addition:
documents that tokenize to **zero** real words are dropped before encoding — an
all-padding row would make every key fully masked in self-attention, which is
undefined for softmax (the same class of bug fixed in
[`09_Attention`](../09_Attention)'s README, handled here by filtering instead of a
finite-mask trick).
