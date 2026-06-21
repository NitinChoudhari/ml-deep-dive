# Extra Trees (Extremely Randomized Trees)

Extra Trees vs Random Forest, head-to-head on the same wine dataset. Both are bagging ensembles of decision trees — the difference is in how each split is chosen.

## What I built
- `extra_trees_vs_random_forest.ipynb` — accuracy and fit-time comparison, 5-fold cross-validation, a multi-seed variance check, and side-by-side feature importance plots

## What I learned
- Random Forest picks the **best** threshold for a feature at each split (searches all candidate thresholds). Extra Trees picks a **random** threshold per feature and only chooses the best feature among those random splits — strictly less search per node
- This extra randomness usually means each individual Extra Trees tree has slightly higher bias, but the ensemble often has **lower variance** — confirmed here by comparing accuracy spread across multiple random seeds
- Extra Trees also typically trains faster than Random Forest since it skips the expensive "find the optimal split point" search — though on a small dataset like this the gap was minor; it matters more at scale
- Cross-validated accuracy between the two was close on this dataset — the real differentiator is robustness/variance, not raw accuracy, which single train/test splits can hide
- Feature importance rankings were broadly similar between the two, since both ultimately rely on the same underlying signal in the data

## Concepts
- Randomized split selection vs greedy best-split search
- Bias-variance trade-off at the ensemble level
- Cross-validation for robust model comparison

## Connections
- Direct comparison with: `14-random-forest/`
