# Naive Bayes

Gaussian Naive Bayes on the breast cancer dataset (30 features, malignant/benign). Applies Bayes' theorem with the "naive" assumption that every feature is conditionally independent given the class.

## What I built
- `naive_bayes_breast_cancer.ipynb` — GaussianNB pipeline, per-class feature distribution plots, ROC curve comparison against Logistic Regression, and a check of how badly the independence assumption is violated

## What I learned
- Bayes' theorem: `P(class | features) ∝ P(class) * P(features | class)`. Naive Bayes makes this tractable by assuming `P(features | class) = Π P(feature_i | class)` — multiply per-feature probabilities instead of modeling their joint distribution
- GaussianNB specifically assumes each feature is normally distributed within each class — plotting per-class histograms shows this is a reasonable approximation for some features here, not perfect for others
- The independence assumption is almost always false in practice — several feature pairs in this dataset had >0.9 correlation — yet the model still performed competitively. The assumption being "wrong" doesn't doom the classifier; it just makes its probability estimates miscalibrated even when the final class prediction is right
- It's extremely fast to train (closed-form, just mean/variance per feature per class) — a good baseline to run before reaching for anything heavier
- It came close to Logistic Regression on ROC-AUC despite being a much simpler model

## Concepts
- Bayes' theorem
- Conditional independence assumption
- Probabilistic classifiers
- ROC curve / AUC as a model comparison tool

## Connections
- Compared against: `06-logistic-regression/`
- Uses entropy/probability ideas from: `01-entropy/`, `05-cross-entropy/`
