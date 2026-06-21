# t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction on the same digits dataset as `21-pca/`, for direct comparison. Built specifically for 2D/3D visualization rather than general-purpose compression.

## What I built
- `tsne_digits.ipynb` — PCA vs t-SNE side-by-side, a `perplexity` sweep, and a same-data-different-seed comparison to show what is and isn't meaningful in a t-SNE plot

## What I learned
- t-SNE preserves **local neighborhood structure** (which points are close to which) rather than global variance like PCA — it produced visibly tighter, more separated clusters per digit class than PCA's 2D projection of the same data
- `perplexity` roughly controls the effective number of neighbors considered per point. Low perplexity over-focuses on very local structure (can fragment a single class into sub-clusters); high perplexity smooths things out and can blur fine distinctions between classes
- **Critical caveat**: absolute distances between clusters and cluster sizes in a t-SNE plot are **not meaningful** — only the grouping/neighborhood relationships are. Re-running with a different random seed reproduced the same groupings but in a different orientation/spacing, confirming this
- t-SNE is stochastic and notably slower than PCA (and, as seen in `23-umap/`, slower than UMAP) — not something you'd run repeatedly inside a training loop
- t-SNE has no `inverse_transform` — unlike PCA, you can't reconstruct an approximation of the original data from the embedding

## Concepts
- Non-linear manifold learning
- Perplexity as a locality hyperparameter
- Stochastic embeddings and reproducibility caveats

## Connections
- Compared against: `21-pca/`, `23-umap/` (same dataset, three-way comparison)
