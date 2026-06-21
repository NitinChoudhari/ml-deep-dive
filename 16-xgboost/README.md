# XGBoost

Gradient boosting on the breast cancer dataset. Unlike Random Forest's independent trees, boosting builds trees **sequentially**, each one targeting the previous ensemble's residual error.

## What I built
- `xgboost_breast_cancer.ipynb` — `XGBClassifier` pipeline, train/test loss curves per boosting round, early stopping, feature importance, and a direct comparison against Random Forest

## What I learned
- Boosting != bagging: each new tree is fit to **correct the mistakes** of the ensemble so far (via gradients of the loss), not trained independently. This is why boosted ensembles can reach lower bias than bagged ones, at the cost of being more overfitting-prone if left unchecked
- Watching train vs test logloss per boosting round makes overfitting visible directly: train loss keeps dropping while test loss eventually flattens or rises
- **Early stopping** (`early_stopping_rounds`) halts training once test loss stops improving for N rounds — a simple, effective regularizer that also saves training time. The "best iteration" was well short of the requested max
- Needed `pip install xgboost` — not part of the base scikit-learn install
- On this dataset, tuned XGBoost edged out a comparable-depth Random Forest, but the gap was modest — boosting's advantage shows up more on harder/larger problems

## Concepts
- Gradient boosting
- Bias-variance trade-off: boosting vs bagging
- Early stopping as regularization
- Train/validation loss curves

## Connections
- Contrast with: `14-random-forest/` (bagging) and `15-extra-trees/`
- Compared against in: `17-lightgbm/`
