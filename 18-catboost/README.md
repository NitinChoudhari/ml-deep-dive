# CatBoost

Gradient boosting built around **native categorical feature support**, demonstrated on a synthetic customer-churn dataset (contract type, payment method, internet service, gender) vs XGBoost which needs manual encoding.

## What I built
- `catboost_churn.ipynb` — synthetic churn dataset with realistic categorical + numeric features, `CatBoostClassifier` trained directly on raw categorical columns, a one-hot-encoded XGBoost comparison, feature importance, and a train/test loss curve

## What I learned
- CatBoost takes a `cat_features` list of column indices and handles encoding internally (using ordered target statistics, not naive one-hot) — no preprocessing step needed
- XGBoost (and most other libraries) require categorical columns to be numeric first. One-hot encoding here expanded 4 categorical columns into many more binary columns — feature count grew noticeably compared to CatBoost's untouched column count
- Despite the extra encoding work, XGBoost's accuracy was comparable on this dataset — CatBoost's real advantage is **convenience and avoiding encoding-related bugs** (e.g., unseen categories at inference, exploding dimensionality with high-cardinality columns), not necessarily a guaranteed accuracy boost on every dataset
- CatBoost's feature importance is reported directly against the original named columns (`contract_type`, `payment_method`, ...) rather than expanded one-hot columns — much easier to interpret
- Needed `pip install catboost`

## Concepts
- Native categorical feature handling
- One-hot encoding and its dimensionality cost
- Gradient boosting (shared with XGBoost/LightGBM)

## Connections
- Compared against: `16-xgboost/`, `17-lightgbm/`
