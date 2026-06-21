# Decision Trees with scikit-learn

Practical application of decision tree classifiers using scikit-learn on the Iris dataset. Demonstrates the full ML pipeline: train/test split, model training, hyperparameter tuning, visualization, and comparison with Random Forest.

## Why sklearn instead of from scratch

After implementing entropy, cross-entropy, Ridge, and Lasso from scratch in NumPy, I made a deliberate engineering decision to use sklearn for decision trees. Here's why:

1. **Decision trees aren't foundational to neural networks** — they're a different paradigm. My implementation-from-scratch energy is better spent on building neural networks, where understanding the internals directly transfers to transformers, CNNs, and LLMs.

2. **Knowing when to use libraries is a senior-engineer skill.** I understand the algorithm deeply (entropy, information gain, Gini impurity — see `01-entropy/`). Implementing it again in 200 lines of NumPy wouldn't teach me anything new at this point.

3. **sklearn's implementation is battle-tested** with edge cases I'd miss in a from-scratch version (numerical stability, efficient sorting, missing value handling).

The goal of this notebook is to demonstrate practical ML engineering: not "can you reinvent the wheel" but "can you apply the right tool correctly."

## What I built

- `decision_tree_iris.ipynb` — full classification pipeline on the Iris dataset including:
  - Stratified train/test split with reproducible random state
  - Decision tree training with both Gini and Entropy criteria
  - Tree visualization showing actual decision rules
  - Hyperparameter tuning across `max_depth` values
  - Confusion matrix and classification report
  - Feature importance analysis
  - Random Forest comparison as a preview of ensemble methods

## What I learned

### Observing the bias-variance tradeoff in real numbers

A fully-grown decision tree (`max_depth=None`) on Iris achieved 100% training accuracy but only ~95% test accuracy. This is the classic overfitting signature — the model memorized training samples (including noise) but failed to fully generalize.

By tuning `max_depth` from None down to 3, I watched the gap between training and test accuracy close. At `max_depth=3`, both accuracies aligned around 95-97% — the sweet spot where the model captures real patterns without memorizing individual data points.

**This is the bias-variance tradeoff observed empirically, not just understood theoretically.** The hyperparameter tuning plot makes it visually obvious.

### Feature importance reveals what the model actually uses

For the Iris dataset, petal length and petal width dominated the feature importance ranking. Sepal dimensions barely contributed. This makes biological sense — petals vary dramatically between Iris species while sepals are more similar.

This kind of interpretability is why decision trees remain popular in regulated industries (medicine, finance, lending) despite being less accurate than ensemble methods. **You can explain a decision tree to a non-technical stakeholder. Try doing that with a neural network.**

### Why stratified splits matter

I used `stratify=y` in `train_test_split` to maintain the same class distribution in training and test sets. Without it, random splits can produce unbalanced sets (e.g., 50 setosa in train, 0 in test), which breaks classification metrics. **Always stratify for classification problems.**

### Random Forest beats a single tree

A Random Forest with 100 trees at `max_depth=3` outperformed a single tree of the same depth by 1-2%. This is the power of **ensemble methods through variance reduction** — averaging predictions across multiple trees trained on different bootstrap samples smooths out individual mistakes.

On larger, noisier datasets the gap is much bigger. Random Forest is rarely worse than a single tree and usually meaningfully better. This is why it's the default "first try" for many practitioners.

## Key hyperparameters to know

| Parameter | What it controls | When to tune |
|-----------|------------------|--------------|
| `criterion` | Gini vs Entropy splitting | Rarely matters (similar results) |
| `max_depth` | How deep the tree can grow | **Critical** — controls overfitting |
| `min_samples_split` | Minimum samples to split a node | Useful for noisy data |
| `min_samples_leaf` | Minimum samples at a leaf | Useful for preventing tiny leaves |
| `max_features` | Features considered per split | Important for Random Forest |
| `random_state` | Reproducibility | Always set for consistent results |

The two parameters that matter most: `max_depth` and `min_samples_leaf`. Tune these first.

## Results summary

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| Decision Tree (max_depth=None) | 100% | ~95% (overfitting) |
| Decision Tree (max_depth=3) | ~97% | ~97% (well-tuned) |
| Decision Tree (max_depth=1) | ~67% | ~67% (underfitting) |
| Random Forest (max_depth=3, n=100) | ~98% | ~97-98% |

The constrained Decision Tree and Random Forest perform similarly on this small clean dataset. On larger/noisier data, Random Forest would pull further ahead.

## When to use decision trees

### Good fit:
- Need to explain the model to non-technical stakeholders
- Mixed feature types (numerical + categorical)
- Don't want to scale features (trees are scale-invariant)
- Want feature importance information
- Quick baseline before trying more complex models

### Bad fit:
- Highly non-linear continuous relationships (use neural nets or gradient boosting)
- Very high-dimensional data
- Single trees alone are usually weaker than ensemble methods on real data
- Streaming / online learning (trees don't easily update with new data)

## Concepts demonstrated

- Train/test split with stratification
- Cross-validation via hyperparameter sweeps
- Bias-variance tradeoff (concrete observation)
- Tree visualization for interpretability
- Confusion matrix and classification report
- Feature importance from trained trees
- Hyperparameter tuning workflow
- Ensemble methods preview (Random Forest)
- When to use libraries vs implement from scratch (engineering judgment)

## Connections

- **Built on:** `01-entropy/` (Gini and entropy are the splitting criteria), `06-logistic-regression/` (confusion matrix, classification report)
- **Leads to:** Random Forests, Gradient Boosting, XGBoost, ensemble methods more broadly
- **Complements:** `09-hierarchical-clustering/` (also a tree-based technique but unsupervised)