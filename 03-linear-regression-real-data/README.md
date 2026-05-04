# Linear Regression on Real-World Data (Feature Scaling)

Same algorithm as `02-Linear Regression`, but applied to noisy real-world housing data (size vs price). This forced me to encounter and solve real ML preprocessing challenges.

## What I built
- `linear_regression_with_scaling.ipynb` — predicts house price from size in sq ft
- Includes train-time visualization of fitted line over scatter plot
- Handles feature scaling explicitly

## What I learned
- Real data is noisy — perfect MSE = 0 is impossible (irreducible error / Bayes error)
- Large input values (X in 1000s) cause gradient explosion → NaN
- **Feature scaling fixes this** — divide X by its max value before training
- After scaling, normal learning rates (0.01) work again
- Slope of fitted line has real-world interpretation: ₹/sq ft

## Real-world problem encountered
- Initial training with raw X (450 to 3000 sq ft) caused gradients to explode
- Fixed by scaling X to [0, 1] range
- Final model fits the data well visually, with sensible coefficients

## Concepts
- Feature scaling / normalization
- Reducible vs irreducible error
- Visual model evaluation
- Coefficient interpretation
