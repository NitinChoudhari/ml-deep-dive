# Multivariate Linear Regression from Scratch

Linear regression generalized to multiple features using NumPy matrix operations. Predicts house price from size, bedroom count, and age.

## What I built
- `multivariate_linear_regression.ipynb` — predicts house price from 3 features
- Vectorized gradient computation using `X.T @ error` (no Python loops)
- Per-feature scaling (each feature normalized to its own max)
- Predicted vs Actual plot for model evaluation
- Loss curve to visualize training convergence

## What I learned
- The same gradient descent loop scales beautifully to multiple features with matrix math
- Each feature must be scaled independently — different scales cause unbalanced gradients
- For multi-feature models, you can't visualize the model directly (lives in higher dimensions)
  - Use **predicted vs actual plots** instead
  - Use **residual plots** to check for systematic bias
  - Use **feature importance bars** to interpret what the model learned
- Coefficient signs match physical intuition:
  - Positive weight for size (bigger = more expensive)
  - Negative weight for age (older = cheaper)

## Math
- Model: `y = X @ w + b` where X is (n_samples, n_features), w is (n_features,)
- Vectorized gradient: `dw = (2/n) * X.T @ error`

## Real-world debugging
- Hit a Jupyter cell-execution-order bug where training ran with unscaled features
- Diagnosed by printing X_scaled stats before training
- Fixed by grouping all preprocessing + training into a single cell
- Lesson: always sanity-check preprocessed data before training

## Concepts
- Vectorized NumPy operations
- Multi-feature gradient descent
- Multiple visualization strategies
- Debugging in notebook environments