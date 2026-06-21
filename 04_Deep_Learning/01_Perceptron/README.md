# Perceptron

The Perceptron is the original neural network unit (Rosenblatt, 1958). It takes a
vector of inputs, computes a weighted sum plus a bias, and passes that through an
activation function to produce a binary output:

```
output = activation(w . x + b)
```

A single perceptron can only learn **linearly separable** decision boundaries. Stack
many of them in layers (with non-linear activations) and you get an ANN — see
[`../02_ANN`](../02_ANN).

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `perceptron_pytorch.ipynb` | PyTorch | `sklearn.datasets.load_breast_cancer` | Binary classification (malignant vs. benign) |
| `perceptron_tensorflow.ipynb` | TensorFlow / Keras | `sklearn.datasets.load_breast_cancer` | Binary classification (malignant vs. benign) |

Both notebooks implement the perceptron as a single linear unit followed by a
sigmoid — in PyTorch that's one `nn.Linear` layer trained with `BCEWithLogitsLoss`;
in Keras it's a single `Dense(1, activation="sigmoid")` layer. Both are trained with
plain SGD and evaluated on held-out test accuracy.
