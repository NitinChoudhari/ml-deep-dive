# PCA (Principal Component Analysis)

Linear dimensionality reduction on the digits dataset (1797 images, 64 pixels each). Finds the directions (principal components) that capture the most variance in the data.

## What I built
- `pca_digits.ipynb` — per-component and cumulative explained variance plots, a 2D projection colored by digit, and a lossy reconstruction from a reduced number of components

## What I learned
- PCA finds orthogonal directions ranked by how much variance they capture — the first component alone captures far more variance than any single original pixel feature
- Plotting cumulative explained variance shows how many components are needed to retain most of the information (e.g., 90%) — far fewer than the original 64 dimensions, confirming the data has a lot of redundancy
- Compressing to just 2D for visualization keeps only a small fraction of total variance, and digit classes overlap substantially — PCA is linear, so it can't unfold non-linear structure in the data (motivating t-SNE/UMAP)
- PCA is **invertible**: `inverse_transform` reconstructs an approximation of the original image from the reduced representation. The reconstruction loses fine detail but the digit is still recognizable — a direct way to see what information was discarded
- Always scale features before PCA — it operates on variance, so unscaled features with larger numeric ranges would dominate the components for no meaningful reason

## Math
- Computes eigenvectors of the (scaled) covariance matrix; eigenvectors = principal component directions, eigenvalues = variance captured by each

## Concepts
- Variance maximization / eigendecomposition
- Explained variance ratio
- Lossy, invertible compression

## Connections
- Compared against: `22-tsne/`, `23-umap/` (same dataset, non-linear alternatives)
