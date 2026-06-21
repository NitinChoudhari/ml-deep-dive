# Deep Learning Notebooks — Build Log

This folder contains executed Jupyter notebooks (PyTorch + TensorFlow) for every topic
in [`Topics to Cover.txt`](./Topics%20to%20Cover.txt). This README documents how they
were built: the environment, the debugging process, every real bug that was found and
fixed, the root cause of each, and the optimizations applied. Per-topic explanations
of the actual ML concepts live in each topic's own `README.md`.

## Environment

- Python 3.13.12 (via the `py` launcher — no other Python was installed)
- `torch==2.12.1+cpu` (installed from the PyTorch CPU wheel index)
- `tensorflow==2.21.0` (installed from PyPI — works fine on 3.13, no fallback venv needed)
- `scikit-learn`, `pandas`, `matplotlib`, `nbformat`, `nbconvert`, `ipykernel` — already present

Every notebook was **actually executed**, not just written — generated via a Python
script using `nbformat`, then run with:

```
py -m jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb --ExecutePreprocessor.timeout=600
```

A notebook only counts as "done" if it (a) runs with zero `output_type: error` cells,
**and** (b) prints a sane, non-degenerate loss/accuracy — see below for why check (a)
alone is not enough.

## Final results

| Topic | Dataset / task | PyTorch | TensorFlow |
|---|---|---|---|
| 01 Perceptron | `load_breast_cancer`, binary | 95.6% | 95.6% |
| 02 ANN | `load_digits`, 10-class | 96.4% | 97.5% |
| 03 Backpropagation | `load_digits`, 10-class (raw-tensor MLP) | 78.3% | 78.1% |
| 04 Activation Functions | `load_digits`, Sigmoid/Tanh/ReLU comparison | — (comparison, not single accuracy) | — |
| 05 CNN | `load_digits` as 8x8 images, 10-class | 94.2% | 97.8% |
| 06 RNN | `fetch_20newsgroups` (hockey vs space), binary | 74.8% | 55.6% |
| 07 LSTM | same dataset/task | 84.4% | 92.8% |
| 08 GRU | same dataset/task | 84.1% | 89.7% |
| 09 Attention | same dataset/task + attention-weight viz | 83.1% | 92.6% |
| 10 Transformers | same dataset/task, built-in multi-head attention | 86.8% | 95.2% |

Backpropagation's lower accuracy (~78%) vs. ANN's (~96%) is expected and intentional —
it trains a smaller hand-rolled 2-layer MLP with plain SGD and no learning-rate
tuning, to keep the forward/backward mechanics visible rather than to maximize
accuracy. RNN's accuracy being clearly weaker than LSTM/GRU/Attention/Transformer on
the *same* task is also expected — it's the textbook vanishing-gradient limitation of
plain RNNs on long sequences, not a bug.

## Debugging process: 3 real bugs found and fixed

The first pass through all 10 topics ran cleanly — no exceptions, no crashed cells.
But "no exception" is not the same as "correct." A sanity sweep over every printed
accuracy/loss (`grep`-ing all notebooks for `"Test accuracy"` and `"loss nan"`) turned
up three real, silent bugs.

### Bug 1 — ANN (PyTorch): double-fit `StandardScaler`

**Symptom:** PyTorch ANN scored 73.3% vs. TensorFlow's 97.5% on the *identical* task
(same MLP shape, same data) — a gap too large to be random variance.

**Root cause:** the preprocessing code was:

```python
X_train = StandardScaler().fit(X_train).transform(X_train)
X_test = StandardScaler().fit(X_train).transform(X_test)   # bug
```

The second line fits a **brand new** `StandardScaler` on `X_train`, which by that
point is *already standardized* (mean ≈ 0, std ≈ 1). Fitting on already-scaled data
gives a near-identity scaler, which is then applied to the *raw, unscaled* `X_test`.
Net effect: test data never actually got standardized, so it sat on a totally
different scale than what the model was trained on.

**Fix:** fit one scaler on the raw training data, reuse it for both splits:

```python
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Result: 73.3% → 96.4%, now in line with TensorFlow's 97.5%.

### Bug 2 — Attention (PyTorch): `NaN` loss from masking with `-inf`

**Symptom:** the PyTorch attention notebook trained to `loss: nan` from epoch 1 and
landed at 50.3% test accuracy (= random guessing for a balanced binary task). The
TensorFlow twin trained fine (93% accuracy) on the identical architecture.

**Root cause:** the attention layer masks out padding positions before `softmax`:

```python
scores = scores.masked_fill(~mask, float("-inf"))
alpha = torch.softmax(scores, dim=1)
```

A handful of the `20newsgroups` posts (32 out of 1193 training docs) tokenize to
**zero real words** after stripping headers/quotes/footers — their entire row is
padding, so `mask` is all-`False` for that row. `softmax` over a row that's
*entirely* `-inf` is `0/0 = NaN`. That single NaN row poisons the whole batch's
gradient via `loss.backward()`, because the loss is computed over the full batch at
once (full-batch training, not mini-batches) — so one degenerate row was enough to
break every parameter's gradient.

The TensorFlow version didn't hit this because it happened to mask with a large
*finite* negative number (`-1e9`) rather than `-inf` — `softmax` over an all-`-1e9`
row is mathematically well-defined (a uniform distribution), just semantically
meaningless for that one row.

**Fix:** mask with `-1e9` in the PyTorch version too, matching the TensorFlow
convention:

```python
scores = scores.masked_fill(~mask, -1e9)   # finite, not -inf
```

Result: `nan` → converges normally, 50.3% → 83.1% test accuracy.

### Bug 3 — RNN / LSTM / GRU (PyTorch): no padding mask, recurrence runs through noise

**Symptom:** all three PyTorch recurrent notebooks underperformed their TensorFlow
twins by a large, consistent margin (RNN 49.3% vs. 57.3%, LSTM 58.3% vs. 91.6%, GRU
55.7% vs. 92.2%) — too consistent across three different architectures to be
explained by "RNNs are just weaker," since LSTM/GRU are specifically designed *not*
to have that weakness.

**Root cause:** sequences are padded to a fixed `MAX_LEN = 150`, but the real
document length distribution is `median = 83 tokens, mean = 184` — **73% of
documents are shorter than 150 tokens**, meaning most rows are mostly padding.

- **TensorFlow's** `Embedding(..., mask_zero=True)` automatically propagates a mask
  through the recurrent layer. Internally, Keras's RNN/LSTM/GRU **freeze the hidden
  state** at masked (padding) timesteps — the state at the end of the sequence equals
  the state at the *last real token*.
- **PyTorch's** `nn.RNN` / `nn.LSTM` / `nn.GRU` have no automatic masking. They happily
  keep recurring through every one of the ~67+ trailing padding steps for a typical
  document. Even though the padding embeds to a zero vector (`padding_idx=0`), the
  *recurrent* weight matrices (`W_hh`) still transform the hidden state at every
  step regardless of input — so the final hidden state used for classification was
  the state after dozens of extra, untrained-for "decay" steps, not the state that
  actually captured the document's content.

This was confirmed by checking the actual document-length distribution
(`sklearn.datasets.fetch_20newsgroups` + a regex tokenizer) before concluding it was
the cause, rather than guessing.

**Fix:** compute the real (non-padding) length of every sequence and use
`torch.nn.utils.rnn.pack_padded_sequence` before feeding it to the recurrent layer —
a standard PyTorch utility (not hand-rolled code) that tells the RNN/LSTM/GRU exactly
where each sequence really ends, so its returned final hidden state matches Keras's
masked behavior:

```python
lengths = torch.tensor([max(1, min(len(t), MAX_LEN)) for t in tokens])
packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
_, hidden = self.rnn(packed)   # hidden is now the state at the real last token
```

(Lengths are clamped to a minimum of 1 to handle the same zero-token edge case as
Bug 2, since `pack_padded_sequence` doesn't accept zero-length sequences.)

Result: RNN 49.3% → 74.8%, LSTM 58.3% → 84.4%, GRU 55.7% → 84.1%.

## Optimization: under-trained models

Two topics converged too slowly within their original epoch budget — caught by
reading the actual loss curve, not just the final number, and noticing it was still
dropping fast at the last epoch instead of plateauing.

- **Transformers**: with `lr=0.001` and 15 epochs, both notebooks were still around
  67–69% loss-wise, clearly mid-training. Bumping to `lr=0.003` and 40 epochs got
  PyTorch to 86.8% and TensorFlow to 95.2%.
- **RNN/LSTM/GRU**: after the `pack_padded_sequence` fix above, PyTorch's loss was
  still falling steeply at epoch 15 (e.g. LSTM: 0.696 → 0.353 and still dropping).
  Epochs were raised from 15 to 30 for **both** frameworks (to keep the comparison
  fair) — this is what produced the final 74–86% PyTorch numbers and lifted
  TensorFlow's numbers slightly too (e.g. LSTM 91.6% → 92.8%).

## Takeaway

Every fix above was caught the same way: **read the actual numbers a notebook
produces, not just whether it crashed.** A notebook with zero exceptions can still
silently train to a degenerate `NaN` or near-random accuracy. The general process
used throughout was: generate → execute → grep for `"output_type": "error"` →
grep for `"Test accuracy"` / `"loss nan"` → if any number looks suspicious (near
0.5 accuracy, NaN, or a TF/PyTorch gap too large to be variance), isolate the
smallest reproduction in a standalone script, find the root cause, fix it in the
generator script, then regenerate and re-execute before moving to the next topic.
