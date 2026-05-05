# Logistic Regression from Scratch

Implementation of binary logistic regression in NumPy — same gradient descent loop as linear regression, with sigmoid activation and binary cross-entropy loss.

## What I built
- `logistic_regression_clean_data.ipynb` — clean, separable data (100% training accuracy)
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

## Concepts covered
- Sigmoid activation function
- Binary cross-entropy loss
- Maximum likelihood estimation (link to MLE perspective)
- Confusion matrix (TP, TN, FP, FN)
- Decision boundary
- Irreducible error / Bayes error
- Why training accuracy < 100% is correct on noisy data
- Calibrated uncertainty in classifiers

## Connections
- Built on: `01-entropy/`, `05-cross-