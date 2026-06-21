# Support Vector Machines (SVM)

SVM decision boundaries and the kernel trick, visualized on synthetic 2D moon-shaped data, plus a tuned multi-class pipeline on the wine dataset.

## What I built
- `svm_kernels_and_margins.ipynb` — kernel comparison (linear/poly/rbf) on non-linearly-separable data, `C` sweep, `gamma` sweep, `GridSearchCV`-tuned multi-class SVM on wine, and a support-vector visualization

## What I learned
- SVM finds the boundary that **maximizes the margin** between classes, not just any separating line
- A **linear kernel cannot separate the two moons** — it draws a straight line through fundamentally curved, interleaved data and gets poor accuracy. The `rbf` kernel projects the data into a higher-dimensional space (without explicitly computing it — the "kernel trick") where it becomes linearly separable
- `C` is the margin-vs-violations trade-off: low `C` allows a wider margin and tolerates some misclassified points (more regularized); high `C` forces the model to classify every training point correctly, narrowing the margin and risking overfitting
- `gamma` (rbf kernel) controls how far a single point's influence reaches: low `gamma` → smooth, far-reaching boundary (can underfit); high `gamma` → tight boundary that wraps individual points (can overfit)
- **Support vectors are the only points that matter** — removing any non-support-vector training point wouldn't change the boundary at all. In this run, only a fraction of the training points ended up as support vectors.
- Tuning `C` and `gamma` together via grid search matters more than tuning either alone — they interact

## Concepts
- Margin maximization
- Kernel trick (linear, polynomial, RBF)
- Hyperparameters `C` and `gamma`
- Support vectors

## Connections
- Contrast with: `11-knn/` — both are distance/geometry-based, but SVM learns a boundary while KNN just memorizes points
