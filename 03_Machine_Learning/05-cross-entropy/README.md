# Cross-Entropy Loss

Building intuition for binary cross-entropy from first principles. Covers why the formula works, the on/off switching trick, asymmetric punishment of confident wrongness, and the 0.693 baseline.

## What I built
- `binaryCrossEntropyLossForOneExampleMachineLearningAngle.ipynb` — derivation, single-sample examples, on/off switching demonstration, loss curve visualization, and multi-sample comparison

## What I learned

**Cross-entropy = average surprise.** It measures how surprised a model is by the truth, given its predictions. 

The formula: loss = -(y * log(p) + (1-y) * log(1-p))

is just two cases compressed into one line:
- When y=1: loss = -log(p)
- When y=0: loss = -log(1-p)

Both reduce to "negative log of the probability the model assigned to the truth" — i.e., the surprise of seeing the truth.

## Key insights

1. **Connection to entropy:** cross-entropy uses the same `-log(p)` "surprise" definition I learned in `01-entropy/`. The chain is unbroken — entropy → cross-entropy → cross-entropy loss.

2. **The 0.693 baseline:** a model that always predicts 0.5 (zero knowledge) has loss = log(2) ≈ 0.693. Any trained classifier must beat this. If your training loss plateaus at 0.69, your model has learned nothing.

3. **Why not MSE for classification?** Cross-entropy explodes for confident wrong predictions (loss → ∞ as p→0 when y=1). MSE only goes up to 1.0. Cross-entropy gives gradient descent a much stronger signal to fix bad predictions.

4. **Numerical stability:** `np.log(0)` is -∞, which crashes training. Always clip predictions to `[eps, 1-eps]` before computing the loss.

## Why this matters for ML

Cross-entropy loss is THE loss function for classification. It powers:
- Logistic regression
- All neural network classifiers
- Multi-class classifiers (categorical cross-entropy)
- LLM training (next-token prediction is just classification over the vocabulary)

Mastering this single concept unlocks understanding of every classifier in ML.

## Connections
- Built on: `01-entropy/`
- Enables: `06-logistic-regression/` (uses this loss)
- Generalizes to: categorical cross-entropy (multi-class), perplexity (language models)