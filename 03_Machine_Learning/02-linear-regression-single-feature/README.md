# Linear Regression from Scratch (Single Feature)

Implementation of linear regression using gradient descent in pure NumPy. Trained on student exam scores vs hours studied.

## What I built
- `linear_regression.ipynb` — predicts exam score from hours studied
- Implements gradient descent loop manually (no sklearn during training)
- Compares against sklearn's LinearRegression as a sanity check

## What I learned
- The 3-step ML training pattern: predict → compute loss → update parameters
- How to derive gradients of MSE with respect to weights and bias
- Why we subtract the gradient (descent, not ascent)
- Learning rate is the most critical hyperparameter
  - Too high → diverges to NaN
  - Too low → trains forever
  - Just right → converges in ~50 epochs
- Linear models extrapolate naively — predicted >100 score for 11 hours studied (impossible in real world)

## Math
- Model: `y = w * x + b`
- Loss: `MSE = (1/n) * Σ(y_pred - y)²`
- Gradients: `dw = (2/n) * Σ(error * x)`, `db = (2/n) * Σ(error)`

## Concepts
- Gradient descent
- MSE loss
- Hyperparameter tuning (learning rate, epochs)
- Extrapolation limitations