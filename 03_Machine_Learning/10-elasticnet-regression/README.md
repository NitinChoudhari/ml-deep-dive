# ElasticNet Regression

ElasticNet combines Ridge's L2 penalty and Lasso's L1 penalty into one model. Applied to the classic diabetes dataset (10 correlated medical features) using scikit-learn.

## What I built
- `elasticnet_regression.ipynb` — full pipeline: correlation analysis, train/test split, Ridge vs Lasso vs ElasticNet comparison, `GridSearchCV` tuning over `alpha` and `l1_ratio`, coefficient visualization, and an `l1_ratio` sweep

## What I learned
- ElasticNet has **two** hyperparameters instead of one: `alpha` (overall regularization strength) and `l1_ratio` (the L1/L2 mix — 0 = pure Ridge, 1 = pure Lasso)
- Lasso alone struggles when features are **correlated** — it tends to arbitrarily pick one feature from a correlated group and zero out the rest, which is unstable
- ElasticNet's L2 component stabilizes that selection: correlated features get similar (shrunk) coefficients instead of one winning arbitrarily
- Sweeping `l1_ratio` at fixed `alpha` shows a direct trade-off: more L1 → more zeroed coefficients (sparser model) but RMSE doesn't always improve — sparsity and accuracy are different objectives
- `GridSearchCV` over a 2D hyperparameter grid is the standard way to tune ElasticNet; tuning either parameter alone misses interactions between them

## Math
- Loss: `MSE + alpha * (l1_ratio * |w|₁ + (1 - l1_ratio) * 0.5 * |w|₂²)`
- `l1_ratio = 1` → Lasso, `l1_ratio = 0` → Ridge

## Concepts
- Regularization (L1 + L2 combined)
- Multicollinearity and feature correlation
- Hyperparameter grid search
- Coefficient shrinkage vs sparsity trade-off

## Connections
- Builds on: `07-ridge-lasso-regression/`
- Same regularization family, generalizes both penalties into one model
