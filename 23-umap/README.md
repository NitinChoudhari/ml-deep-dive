# UMAP (Uniform Manifold Approximation and Projection)

Completes the three-way dimensionality reduction comparison (`21-pca/`, `22-tsne/`, this notebook) on the same digits dataset. Non-linear like t-SNE, but built on manifold-learning/topology theory and runs noticeably faster.

## What I built
- `umap_digits.ipynb` — PCA/t-SNE/UMAP side-by-side with timing, and sweeps over UMAP's two key hyperparameters (`n_neighbors`, `min_dist`)

## What I learned
- UMAP produced a 2D embedding visually comparable in cluster separation to t-SNE, but ran significantly faster — confirmed by timing both on the exact same data, which is the main practical reason UMAP has largely replaced t-SNE for exploratory visualization on larger datasets
- `n_neighbors` is UMAP's analog to t-SNE's `perplexity`: small values emphasize fine local structure (can fragment classes), large values pull in more global structure (smoother, more connected layout)
- `min_dist` controls how tightly points are allowed to pack together in the embedding — low values produce dense, tight clusters; high values spread points out more evenly, trading visual cluster tightness for a layout that better reflects local density variation
- Needed `pip install umap-learn` — not part of base scikit-learn
- Like t-SNE, UMAP embeddings are not invertible and absolute distances aren't fully meaningful — but UMAP is generally considered better at preserving *some* global structure (relative positioning between distant clusters) than t-SNE does

## Concepts
- Manifold learning (topological approach)
- `n_neighbors` / `min_dist` hyperparameters
- Speed vs t-SNE for large-scale visualization

## Connections
- Three-way comparison with: `21-pca/`, `22-tsne/`
