# Activation Functions

An activation function is applied after each linear layer to introduce non-linearity —
without it, stacking layers ([`ANN`](../02_ANN)) would collapse into one big linear
function, no matter how many layers you add.

Common choices:

| Function | Formula | Range | Notes |
|---|---|---|---|
| Sigmoid | `1 / (1 + e^-x)` | (0, 1) | Saturates for large \|x\|, slow gradients deep in a network |
| Tanh | `(e^x - e^-x) / (e^x + e^-x)` | (-1, 1) | Zero-centered version of Sigmoid, same saturation issue |
| ReLU | `max(0, x)` | [0, ∞) | Cheap, doesn't saturate for x > 0, the modern default |

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `activation_functions_pytorch.ipynb` | PyTorch | `sklearn.datasets.load_digits` | 10-class digit classification |
| `activation_functions_tensorflow.ipynb` | TensorFlow / Keras | `sklearn.datasets.load_digits` | 10-class digit classification |

Each notebook first plots the three curves, then trains the **same** 64->32->10 MLP
three times — swapping only the activation (Sigmoid / Tanh / ReLU) — and plots all
three training-loss curves together so you can directly compare how the choice of
activation affects convergence.
