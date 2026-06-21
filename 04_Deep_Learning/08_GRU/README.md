# GRU (Gated Recurrent Unit)

A GRU is a streamlined alternative to the [`LSTM`](../07_LSTM). It merges the forget
and input gates into one **update gate**, adds a **reset gate**, and drops the
separate cell state — using only a single hidden state. Fewer parameters than an
LSTM, often trains faster, and frequently reaches comparable accuracy.

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `gru_pytorch.ipynb` | PyTorch | `sklearn.datasets.fetch_20newsgroups` (`rec.sport.hockey` vs `sci.space`) | Binary text classification |
| `gru_tensorflow.ipynb` | TensorFlow / Keras | same | same |

Identical pipeline and task to [`06_RNN`](../06_RNN) and [`07_LSTM`](../07_LSTM) —
same tokenizer, same 5000-word vocabulary, same 150-token sequences — with only the
recurrent layer swapped: `nn.LSTM` -> `nn.GRU` (PyTorch), `LSTM` -> `GRU` (Keras).
