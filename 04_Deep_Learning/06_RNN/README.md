# RNN (Recurrent Neural Network)

An RNN processes a sequence one element at a time, maintaining a **hidden state**
that gets updated at every step and carried forward — so the network's interpretation
of token *t* depends on everything it has seen before it. This makes RNNs a natural
fit for text, where word order and context matter, unlike the fixed-size inputs used
by [`ANN`](../02_ANN)/[`CNN`](../05_CNN).

```
h_0 = 0
h_t = activation(W_x . x_t + W_h . h_{t-1} + b)
output = f(h_T)   # final hidden state used for classification
```

Plain RNNs struggle to retain information over long sequences (vanishing gradients) —
that's what [`LSTM`](../07_LSTM) and [`GRU`](../08_GRU) fix.

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `rnn_pytorch.ipynb` | PyTorch | `sklearn.datasets.fetch_20newsgroups` (`rec.sport.hockey` vs `sci.space`) | Binary text classification |
| `rnn_tensorflow.ipynb` | TensorFlow / Keras | same | same |

Pipeline used in both (and reused unchanged in `07_LSTM`, `08_GRU`, `09_Attention`,
`10_Transformers` — only the recurrent/attention layer changes):

1. Regex word tokenizer (`[a-z]+`, lowercased).
2. A 5000-word vocabulary built from the **training** set only, with `<pad>`/`<unk>`.
3. Every document encoded as a fixed-length (150 token) id sequence.
4. `Embedding -> RNN -> Dense(1, sigmoid)`.
