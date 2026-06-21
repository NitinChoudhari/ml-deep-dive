# Bias-Variance Tradeoff

Made concrete with polynomial regression at increasing degree on noisy synthetic data (a sine wave), including a proper bias²/variance decomposition via repeated resampling — not just the usual hand-wavy explanation.

## What I built
- `bias_variance_tradeoff.ipynb` — underfit/good-fit/overfit comparison at degree 1/4/15, a full degree sweep showing the classic U-shaped test error curve, and a resampling-based bias²+variance decomposition with direct visualization of both terms

## What I learned
- **Underfitting (high bias)**: degree-1 (linear) model is too simple to capture the curve — both train and test error are high, and they're close to each other
- **Overfitting (high variance)**: degree-15 model chases individual noisy training points — train error keeps dropping but test error rises sharply. The fitted curve wiggles in ways that have nothing to do with the true underlying function
- The train-vs-test error plot across degrees is the textbook bias-variance curve: train error monotonically decreases with complexity, test error decreases then increases, and the minimum of the test curve is the actual sweet spot — not the lowest train error
- **The resampling decomposition makes "bias" and "variance" literal, not metaphorical**: refitting the same-degree model on many different noisy resamples of the data and looking at the spread of predictions at each point. Low-degree models gave nearly identical (low variance) but consistently wrong (high bias) curves across resamples. High-degree models gave wildly different curves per resample (high variance) even though their *average* prediction wasn't bad
- Bias² + variance (plus irreducible noise, not modeled here) is the actual decomposition of expected test error — both terms are real, measurable quantities, not just intuition

## Concepts
- Underfitting vs overfitting
- Bias²/variance decomposition via resampling
- Model complexity as the controlling knob

## Connections
- Same underlying idea seen in: `11-knn/` (K controls bias/variance), `07-ridge-lasso-regression/`, `10-elasticnet-regression/` (regularization controls bias/variance)
