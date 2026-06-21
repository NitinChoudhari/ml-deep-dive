# Backpropagation

Backpropagation is the algorithm used to train an [`ANN`](../02_ANN). After a forward
pass produces a prediction and a loss, backprop applies the chain rule layer-by-layer,
from the output back to the input, to compute how much each weight contributed to the
error (its gradient). An optimizer then nudges every weight a small step opposite its
gradient.

```
forward:   x -> Linear1 -> activation -> Linear2 -> logits -> loss
backward:  d(loss)/d(logits) -> ... -> d(loss)/d(Linear2 weights)
                                -> ... -> d(loss)/d(Linear1 weights)
```

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `backpropagation_pytorch.ipynb` | PyTorch | `sklearn.datasets.load_digits` | 10-class digit classification |
| `backpropagation_tensorflow.ipynb` | TensorFlow | `sklearn.datasets.load_digits` | 10-class digit classification |

Unlike the [`ANN`](../02_ANN) notebooks, these build the 64→32→10 MLP from **raw
tensors** (no `nn.Module`, no `model.fit()`), so the mechanics are visible:

- PyTorch: manual `W1, b1, W2, b2` leaf tensors, `loss.backward()` fills `.grad` for
  each, then a manual gradient-descent update (`p -= lr * p.grad`) — no optimizer object.
- TensorFlow: manual `tf.Variable`s, `tf.GradientTape` records the forward pass and
  `tape.gradient(loss, params)` returns the gradients, then a manual `assign_sub` update.

Each notebook prints the total gradient norm every few epochs and plots loss + gradient
norm over training, so you can see backprop actually doing work step by step.
