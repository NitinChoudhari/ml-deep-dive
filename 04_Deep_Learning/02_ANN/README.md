# ANN (Artificial Neural Network / MLP)

An ANN stacks multiple layers of perceptron-like units ("neurons") with non-linear
activation functions between them. This lets the network approximate non-linear
decision boundaries — something a single [`Perceptron`](../01_Perceptron) can't do.

A basic ANN (a.k.a. Multi-Layer Perceptron) looks like:

```
input -> Linear -> activation -> Linear -> activation -> ... -> Linear -> output
```

Training it requires propagating the error backward through every layer — see
[`../03_Backpropagation`](../03_Backpropagation).

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `ann_pytorch.ipynb` | PyTorch | `sklearn.datasets.load_digits` | 10-class digit classification |
| `ann_tensorflow.ipynb` | TensorFlow / Keras | `sklearn.datasets.load_digits` | 10-class digit classification |

Both notebooks use a small feedforward network with two hidden layers
(64 -> 64 -> 32 -> 10) trained with the Adam optimizer and cross-entropy loss on the
real 8x8 handwritten-digit images from `sklearn`.
