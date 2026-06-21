# Attention

An [`LSTM`](../07_LSTM)/[`GRU`](../08_GRU) typically uses only its **final** hidden
state to make a decision, throwing away the rest. Attention instead keeps every
hidden state and learns a weight for each one — so the model decides, per example,
which words mattered most. It's also interpretable: you can plot the weights and see
exactly what the model focused on.

This is **additive (Bahdanau-style) attention** used as a pooling mechanism:

```
score_t = v . tanh(W . h_t)              # one score per timestep
alpha   = softmax(score)                 # normalize into weights that sum to 1
context = sum_t(alpha_t * h_t)           # weighted sum of all hidden states
output  = Dense(context)
```

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `attention_pytorch.ipynb` | PyTorch | `sklearn.datasets.fetch_20newsgroups` (`rec.sport.hockey` vs `sci.space`) | Binary text classification + attention-weight visualization |
| `attention_tensorflow.ipynb` | TensorFlow / Keras | same | same |

Same pipeline as [`06_RNN`](../06_RNN)/[`07_LSTM`](../07_LSTM)/[`08_GRU`](../08_GRU),
but the LSTM now returns a hidden state for **every** token (not just the last one),
and a small additive-attention layer learns to weight them before the final
classifier. Each notebook ends by plotting the attention weight assigned to every
word in one test example.

**Gotcha hit while building this:** masking padded positions with `float("-inf")`
before `softmax` causes `NaN` whenever a row is *entirely* padding (a handful of
these newsgroup posts tokenize to zero real words after stripping headers/quotes) —
`softmax([-inf, ..., -inf])` is 0/0. Both notebooks mask with a large finite negative
number (`-1e9`) instead, which gives a well-defined (uniform) distribution in that
edge case.
