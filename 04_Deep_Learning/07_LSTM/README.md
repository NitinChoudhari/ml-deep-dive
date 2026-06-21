# LSTM (Long Short-Term Memory)

An LSTM is an [`RNN`](../06_RNN) variant designed to remember information over long
sequences. It keeps a separate **cell state** (long-term memory) alongside the hidden
state, and uses three learned gates to control it:

- **Forget gate** — what to drop from the cell state
- **Input gate** — what new information to add
- **Output gate** — what to expose as the hidden state at this step

This gating mechanism avoids the vanishing-gradient problem that limits plain RNNs on
long sequences.

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `lstm_pytorch.ipynb` | PyTorch | `sklearn.datasets.fetch_20newsgroups` (`rec.sport.hockey` vs `sci.space`) | Binary text classification |
| `lstm_tensorflow.ipynb` | TensorFlow / Keras | same | same |

Identical pipeline and task to [`06_RNN`](../06_RNN) — same tokenizer, same
5000-word vocabulary, same 150-token sequences — with only the recurrent layer
swapped: `nn.RNN` -> `nn.LSTM` (PyTorch), `SimpleRNN` -> `LSTM` (Keras). Compare the
loss curves against `06_RNN` directly.
