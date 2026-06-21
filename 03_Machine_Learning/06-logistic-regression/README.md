# Logistic Regression from Scratch

Implementation of binary logistic regression in NumPy — same gradient descent loop as linear regression, with sigmoid activation and binary cross-entropy loss.

## What I built
- `logistic_regression_prefect_data.ipynb` — clean, separable data (100% training accuracy)
- `logistic_regression_noisy_data.ipynb` — realistic noisy data with outliers, evaluated with confusion matrix

## What I learned

### The structural insight
Logistic regression is **linear regression + sigmoid + cross-entropy loss**. The training loop, gradient form, and parameter update rule are identical to linear regression. Only two changes:
1. Apply sigmoid to predictions: `y_pred = sigmoid(w*X + b)`
2. Use binary cross-entropy as the loss function (instead of MSE)

This unifies linear regression, logistic regression, and the building blocks of neural networks under one mental model.

### Why cross-entropy + sigmoid pair beautifully
The gradient of cross-entropy w.r.t. weights simplifies to `(1/n) * Σ(error * x)` — the same form as linear regression's gradient (minus the factor of 2). This happens because the `sigmoid * (1 - sigmoid)` term in the chain rule cancels with the same term in cross-entropy's derivative. This is not coincidence — cross-entropy was specifically chosen as the loss for logistic regression to avoid the vanishing gradient problem that MSE + sigmoid would cause.

### Irreducible error is real
On clean data, loss converged to ~0.01. On noisy data (with 2 outlier students), loss got STUCK at ~0.28 no matter how long I trained. This is the **Bayes error / irreducible error** — the noise floor of the dataset. No model can predict random noise. Training longer would only overfit.

### 100% training accuracy is often a red flag
On noisy data, my model got 90% accuracy. The 10% misclassified were the noisy points (a student who studied 3.5 hours and passed, one who studied 6 hours and failed). A model that achieved 100% on this data would be overfitting.

### Calibrated uncertainty
The sigmoid curve was steep on clean data (almost a step function) but gentler on noisy data. In noisy regions of feature space, the model honestly expresses uncertainty (probability ~0.5) rather than committing to confident wrong predictions. This is the value of probabilistic classifiers over hard threshold models.

## Math
- Model: `y_pred = sigmoid(w*X + b)`
- Sigmoid: `σ(z) = 1 / (1 + e^(-z))`
- Loss: `L = -(1/n) * Σ [y*log(y_pred) + (1-y)*log(1-y_pred)]`
- Gradients: `dw = (1/n) * X.T @ (y_pred - y)`, `db = (1/n) * Σ(y_pred - y)`
- Decision boundary: `x = -b/w` (where sigmoid output = 0.5)

## Confusion matrix on noisy data
| | Predicted PASS | Predicted FAIL |
|---|---|---|
| **Actual PASS** | TP = 10 | FN = 1 |
| **Actual FAIL** | FP = 1 | TN = 8 |

Accuracy = 90%, with errors only on the genuinely noisy outlier points.
## Evaluation: ROC curve

The default decision threshold (0.5) is arbitrary. By varying the threshold from 0 to 1, I traced out the **ROC (Receiver Operating Characteristic) curve** — a plot of True Positive Rate vs False Positive Rate at every threshold.

I implemented ROC two ways:

1. **From scratch** — manually computed TPR and FPR at thresholds [0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.99]
2. **Using sklearn's `roc_curve`** — auto-selects all useful threshold transitions for a smoother curve

The two implementations agree (manual points overlay perfectly on sklearn's curve), validating the from-scratch math.

### Why ROC matters

Different applications demand different thresholds:

- **Cancer screening:** lower the threshold to maximize TPR (don't miss cancer cases, accept some false alarms)
- **Spam filter:** raise the threshold to minimize FPR (don't block legitimate emails, accept some spam slipping through)

ROC lets you tune the model to match the business need **without retraining**. This single insight has earned me a deep appreciation for probabilistic classifiers.

## Evaluation: AUC (Area Under Curve)

The ROC curve is great visually, but AUC summarizes it as a single number between 0 and 1.

**My model achieved AUC = 0.9495** — excellent for a noisy dataset.

AUC has a beautiful interpretation: it's **the probability that a randomly chosen positive example will be ranked higher than a randomly chosen negative example**. So AUC = 0.9495 means: "If I pick a random student who passed and a random student who failed, my model assigns a higher pass-probability to the actual passer 94.95% of the time."

I computed AUC using **sklearn's `roc_auc_score`** (production approach)

All three methods agreed exactly at 0.9495, confirming I understand both the math and the interpretation.

### AUC interpretation cheatsheet

| AUC          | Interpretation        |
|--------------|------------------------|
| 1.0          | Perfect classifier     |
| 0.95         | Excellent              |
| 0.85         | Good                   |
| 0.75         | Acceptable             |
| 0.65         | Weak                   |
| 0.5          | Random (coin flip)     |
| < 0.5        | Worse than random      |

### Why AUC is preferred over accuracy

- **Threshold-independent** — describes classifier performance across all thresholds, not just at 0.5
- **Insensitive to class imbalance** — works well even when classes are skewed
- **Probabilistic interpretation** — easy to explain to non-technical stakeholders
- **Comparable across models** — direct numerical comparison between different classifiers

## Concepts covered

- Sigmoid activation function
- Binary cross-entropy loss
- Maximum likelihood estimation (link to MLE perspective)
- Confusion matrix (TP, TN, FP, FN)
- Decision boundary
- Irreducible error / Bayes error
- Why training accuracy < 100% is correct on noisy data
- Calibrated uncertainty in classifiers
- ROC curve construction and interpretation
- AUC computation
- Threshold tuning for application-specific tradeoffs (cancer screening vs spam filter)

## Connections

- **Built on:** `01-entropy/`, `05-cross-entropy/`, `02-linear-regression-single-feature/`
- **Enables:** multivariate logistic regression, neural networks (a single neuron = logistic regression with any activation)