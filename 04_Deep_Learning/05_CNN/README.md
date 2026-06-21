# CNN (Convolutional Neural Network)

A CNN is built for grid-like data (images). Instead of every neuron connecting to
every pixel (as a plain [`ANN`](../02_ANN) would), a **convolutional** layer slides a
small learned filter across the image, reusing the same weights at every position.
This exploits spatial structure (nearby pixels are related) and uses far fewer
parameters. **Pooling** layers then downsample the result, keeping the strongest
signal while shrinking the spatial size.

```
image -> Conv -> activation -> Pool -> Conv -> activation -> Pool -> Flatten -> Dense -> output
```

## What's in this folder

| File | Framework | Dataset | Task |
|---|---|---|---|
| `cnn_pytorch.ipynb` | PyTorch | `sklearn.datasets.load_digits` (as 8x8 images) | 10-class digit classification |
| `cnn_tensorflow.ipynb` | TensorFlow / Keras | `sklearn.datasets.load_digits` (as 8x8 images) | 10-class digit classification |

Unlike [`02_ANN`](../02_ANN), which flattens each digit into a 64-length vector, these
notebooks keep the digits as real 8x8 single-channel images and run them through two
`Conv -> MaxPool` blocks (1->8->16 channels) before a final dense classifier.
