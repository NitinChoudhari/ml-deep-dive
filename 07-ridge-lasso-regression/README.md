# Ridge & Lasso Regression from Scratch

Implementation of L2 (Ridge) and L1 (Lasso) regularization in NumPy by adding a single term to linear regression's gradient. Same training loop, same gradient descent, just one extra line.

## What I built

- `ridge_lasso_regression.ipynb` — Linear, Ridge, and Lasso compared side-by-side on the housing dataset (size → price)
- Lambda sweep experiments showing how regularization strength affects weight shrinkage
- Visual comparison of all three regression lines on the same plot

## What I learned

### Regularization is just one extra term in the gradient

Both Ridge and Lasso start from the same place — your existing linear regression code. The only change is in the gradient computation:

| Method | Loss function | Gradient term added |
|--------|---------------|---------------------|
| Linear | `MSE` | nothing |
| Ridge (L2) | `MSE + λ·Σw²` | `+ 2λw` |
| Lasso (L1) | `MSE + λ·\|w\|` | `+ λ·sign(w)` |

The training loop is identical. Just one extra term. That's the whole implementation difference.

### Why the gradient terms look different from the loss terms

The loss has `w²` (Ridge) and `|w|` (Lasso). The gradient has `2w` (Ridge) and `sign(w)` (Lasso). This is just calculus:

- Derivative of `λ·w²` is `2λw` (chain rule on the square)
- Derivative of `λ·|w|` is `λ·sign(w)` (derivative of absolute value is +1 or -1 depending on sign)

Same pattern as the "factor of 2" we saw in MSE's gradient — the constant in the gradient is always traceable to differentiating the loss.

### Why Ridge shrinks weights smoothly but Lasso can drive them to EXACTLY zero

This is the key conceptual insight, and you can see it in the gradient pull strength:

**Ridge gradient pull:** `2λw` — proportional to w itself
- When w is large → pull is strong
- When w is small → pull is weak
- As w → 0 → pull → 0
- Result: weights shrink asymptotically toward 0 but never reach it

**Lasso gradient pull:** `λ·sign(w)` — constant magnitude regardless of w
- Pull is always λ in size, regardless of w
- Even when w is tiny, the pull stays at full strength
- This constant force can push weights through zero and snap them there
- Result: weights can land at EXACTLY 0 (feature elimination)

This is why Lasso does automatic feature selection — irrelevant features get their weights pinned to zero — while Ridge just shrinks all weights smoothly.

### Why bias is NOT regularized

In the implementation, we only add the regularization term to the weight gradient, not the bias gradient:

```python
dw = (2/n) * np.sum(error * X) + 2 * lambda_reg * w   # regularized
db = (2/n) * np.sum(error)                             # NOT regularized

The bias represents the baseline prediction (when all features are zero). Penalizing the bias would force predictions toward zero regardless of the data, which destroys the model's ability to fit any non-zero target. Standard practice in all ML libraries.

### λ doesn't mean the same thing in Ridge vs Lasso

Spotted this myself during the experiment: at the same λ value, Ridge shrunk weights way more aggressively than Lasso.

**Why:** Ridge's penalty grows quadratically with weight magnitude (`λ·w²`), so even small λ has strong pull when w is large. Lasso's penalty grows linearly (`λ·|w|`), so the same λ produces a milder effect.

**Practical consequence:** Always tune λ separately for each method. Never assume "λ=10" means the same thing for both. In sklearn, `Ridge(alpha=10)` and `Lasso(alpha=10)` produce different shrinkage levels.

### What I observed in the lambda sweep

For housing data (single feature, size → price), starting from linear's `w = 89.5`:

**Ridge:**
- λ=0.001 → w ≈ 89.4 (almost no effect)
- λ=0.1 → w ≈ 80.7 (mild)
- λ=1 → w ≈ 38.8 (strong)
- λ=10 → w ≈ 3.95 (crushed near zero)

**Lasso:**
- λ=1 → w ≈ 88.5 (mild)
- λ=10 → w ≈ 78.7 (visible shrinkage)
- λ=50 → w ≈ 38.9 (heavy)
- λ=100 → w ≈ 0.0 (EXACTLY zero — Lasso magic)

You can literally watch the regression line get flatter and flatter as λ increases. At very large λ, the line becomes horizontal at `y = mean(y)`.

## When to use which

| Method | Use when... |
|--------|-------------|
| **Plain linear** | Small clean dataset, no feature selection needed |
| **Ridge (L2)** | Default choice. All features matter somewhat. Features may be correlated (multicollinearity). Want stable predictions. |
| **Lasso (L1)** | Many features suspected to be noise/irrelevant. Want automatic feature selection. Need a sparse, interpretable model. |
| **Elastic Net** | Want feature selection AND have correlated features. Combines L1 + L2. |

## Decision boundary intuition

Geometrically, the difference between Ridge and Lasso comes from the shape of their constraint regions:

- **Ridge:** circle (sphere in higher dimensions) — smooth boundary
- **Lasso:** diamond (rotated square) — has CORNERS at the axes (where one weight = 0)

The MSE loss surface, expanding outward, is more likely to first touch a corner of the diamond than a smooth point on a circle. **Hitting a corner = one weight becomes exactly zero.** That's the geometric reason for Lasso's feature selection behavior.

## Math reference
Plain Linear Regression
Loss:     L = (1/n) Σ (y_pred - y)²
Gradient: dw = (2/n) Σ (error · X)

Ridge Regression (L2)
Loss:     L = (1/n) Σ (y_pred - y)² + λ Σ w²
Gradient: dw = (2/n) Σ (error · X) + 2λw

Lasso Regression (L1)
Loss:     L = (1/n) Σ (y_pred - y)² + λ Σ |w|
Gradient: dw = (2/n) Σ (error · X) + λ · sign(w)

## Concepts covered

- L1 vs L2 regularization (mathematical difference)
- Why Ridge shrinks but doesn't eliminate weights
- Why Lasso performs automatic feature selection
- The bias-variance tradeoff and how λ controls it
- Why λ scales differently between Ridge and Lasso
- Why bias terms are not regularized
- Geometric interpretation of L1 (diamond) vs L2 (circle) constraints
- How regularization solves overfitting and multicollinearity

## Connections

- **Built on:** `02-linear-regression-single-feature/`, `03-linear-regression-real-data/`, `04-multivariate-linear-regression/`
- **Same idea applies to:** logistic regression (L1/L2 regularized logistic regression is a standard sklearn option)
- **Foundation for:** Elastic Net, neural network weight decay (which is just L2 regularization on NN weights)

## Common interview questions

**Q: Why does Lasso do feature selection but Ridge doesn't?**
Lasso's gradient has constant magnitude (`λ·sign(w)`), pulling weights toward zero with full force regardless of w's size. Ridge's gradient (`2λw`) weakens as w shrinks, so weights asymptotically approach but never reach zero. Geometrically, Lasso's diamond constraint region has corners on the axes — solutions often land there, with one weight = 0.

**Q: When would you use Ridge over Lasso?**
When all features contribute meaningfully and you want stability over sparsity. Ridge handles correlated features gracefully (spreading importance), while Lasso arbitrarily picks one and zeros the rest. Ridge is the default for most general-purpose linear models.

**Q: What does the regularization parameter λ control?**
The trade-off between fitting the data (low MSE) and keeping weights small (low complexity). λ=0 means no regularization (potential overfitting). λ→∞ means weights forced to zero (underfitting). Optimal λ is found via cross-validation. Tuning λ is one of the most fundamental skills in classical ML.

**Q: Why don't we regularize the bias?**
The bias represents the baseline prediction when features are zero. Regularizing it would force predictions toward zero regardless of the actual data, breaking the model's ability to fit non-zero targets.