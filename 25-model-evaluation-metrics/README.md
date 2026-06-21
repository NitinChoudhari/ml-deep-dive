# Model Evaluation Metrics

Classification metrics on an artificially imbalanced breast-cancer dataset (to expose why accuracy alone lies), and regression metrics on the diabetes dataset.

## What I built
- `model_evaluation_metrics.ipynb` — accuracy/precision/recall/F1 on imbalanced data, confusion matrix, ROC curve vs Precision-Recall curve, and MAE/MSE/RMSE/R² with a residual plot on a regression task

## What I learned
- On an artificially imbalanced split, a model that **always predicts the majority class** would already score high accuracy — close enough to the trained model's actual accuracy that accuracy alone tells you almost nothing useful here
- **Precision** = of everything predicted positive, how much was actually positive. **Recall** = of everything actually positive, how much did we catch. They trade off against each other and which one matters depends entirely on the cost of false positives vs false negatives for the specific problem
- **ROC-AUC can look deceptively good under heavy class imbalance** because it credits the model for correctly identifying the (easy, abundant) negative class. **PR-AUC** ignores true negatives entirely and is the more honest metric when the positive class is rare and what you actually care about
- For regression: **MAE** is the average absolute error (robust to outliers, easy to interpret in the target's units). **MSE/RMSE** square the error first, so they penalize large errors disproportionately more — visualized directly by plotting `|error|` vs `error²` side by side
- **R²** measures the fraction of variance explained relative to just predicting the mean — useful for a single "is this model any good at all" number, but it doesn't say anything about where the errors are concentrated
- A **residual plot** (residual vs predicted value) reveals patterns a single summary metric can't — e.g. the model being systematically worse for certain prediction ranges

## Concepts
- Precision/recall/F1 trade-off
- ROC vs Precision-Recall curves under class imbalance
- MAE vs MSE/RMSE sensitivity to outliers
- Residual analysis

## Connections
- Builds on: `05-cross-entropy/` (the loss these classifiers were trained to minimize)
- Used throughout: every classification/regression notebook in this repo implicitly relies on these metrics
