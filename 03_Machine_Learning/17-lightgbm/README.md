# LightGBM

Gradient boosting like XGBoost, but grown **leaf-wise** instead of **level-wise**, on the breast cancer dataset.

## What I built
- `lightgbm_breast_cancer.ipynb` — `LGBMClassifier` pipeline, `num_leaves` sweep, fit-time comparison against XGBoost, feature importance, and accuracy-vs-`n_estimators` comparison

## What I learned
- **Level-wise growth** (XGBoost's default) splits every leaf at the current depth before going deeper — balanced but can waste splits on low-value leaves. **Leaf-wise growth** (LightGBM) always splits whichever leaf reduces loss the most, regardless of depth — can converge faster but risks deeper, more overfit-prone trees on small data, which is why `num_leaves` needs tuning rather than just `max_depth`
- `num_leaves` is LightGBM's primary complexity knob since leaf-wise trees don't grow uniformly — accuracy improved then plateaued/dipped as leaves increased, the leaf-wise version of the usual bias-variance curve
- LightGBM uses histogram-based binning for finding splits (bucket continuous features into discrete bins) which is what makes it fast on large/wide datasets — on this small dataset the speed difference vs XGBoost was minor, as expected, since the optimization pays off at scale, not on toy data
- Needed `pip install lightgbm`
- Final accuracy was comparable to XGBoost — the meaningful difference here is training mechanics and speed at scale, not raw accuracy on a small dataset

## Concepts
- Leaf-wise vs level-wise tree growth
- Histogram-based split finding
- `num_leaves` as a complexity hyperparameter

## Connections
- Direct comparison with: `16-xgboost/`
- Compared against in: `18-catboost/` (conceptually, all three are gradient boosting variants)
