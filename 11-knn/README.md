# K-Nearest Neighbors (KNN)

KNN classification on the wine dataset (13 chemical features, 3 cultivars). KNN has no training phase — it memorizes the data and classifies new points by majority vote among the K closest neighbors.

## What I built
- `knn_wine_classification.ipynb` — scaled vs unscaled comparison, elbow-method K sweep, final evaluation, and a 2-feature decision boundary plot

## What I learned
- KNN is a **lazy learner** — "training" is just storing the data; all the work happens at prediction time
- It's purely distance-based, so **feature scaling is mandatory** — a feature ranging 0–1000 will dominate the Euclidean distance over one ranging 0–1, regardless of actual importance. Scaling improved accuracy notably in this run.
- Small K → low bias, high variance (overfits to noise, decision boundary gets jagged)
- Large K → high bias, low variance (boundary gets smoother, can underfit and blur class edges)
- The accuracy-vs-K curve is a direct visual of the bias-variance trade-off — pick K where test accuracy peaks before it starts degrading
- Prediction is O(n) per query (must compute distance to every training point) — doesn't scale to huge datasets without index structures (KD-trees, ball trees)

## Concepts
- Distance metrics (Euclidean)
- Feature scaling for distance-based models
- Bias-variance trade-off via hyperparameter K
- Decision boundary visualization

## Connections
- Contrast with: `06-logistic-regression/` (parametric, learns a boundary) vs KNN (non-parametric, memorizes data)
