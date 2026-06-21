# Random Forest

An ensemble of decision trees on the wine dataset, each trained on a bootstrap sample with random feature subsets per split, combined by majority vote. Direct follow-up to the single-tree pipeline in `08-decision-trees-sklearn/`.

## What I built
- `random_forest_wine.ipynb` — single tree vs forest comparison, OOB score, accuracy-vs-`n_estimators` curve, feature importance, and a look at individual trees disagreeing while the forest vote stays stable

## What I learned
- A single unconstrained decision tree memorizes the training set (near-100% train accuracy) but generalizes worse than the forest — classic overfitting that bagging fixes
- **Bagging** = train each tree on a random bootstrap sample (with replacement) of the data, and at each split only consider a random subset of features. This decorrelates the trees so their errors don't all point the same direction
- **OOB (out-of-bag) score** is a free validation estimate: since each tree only sees ~63% of the data (bootstrap sampling), the other ~37% can be used to validate that tree without a separate holdout set
- Accuracy vs `n_estimators` plateaus quickly — more trees reduce variance but eventually stop helping; it's a "more is safer, not always better" hyperparameter, unlike a typical bias-controlling one
- Feature importance is now an **average across all trees**, generally more stable than a single tree's importance

## Concepts
- Bootstrap aggregation (bagging)
- Out-of-bag (OOB) estimation
- Ensemble variance reduction
- Feature importance averaging

## Connections
- Builds on: `08-decision-trees-sklearn/`
- Contrast with: `15-extra-trees/` (randomized splits instead of best splits)
