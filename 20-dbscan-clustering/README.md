# DBSCAN (Density-Based Clustering)

Density-based clustering on moon-shaped data with injected outliers, contrasted directly against KMeans.

## What I built
- `dbscan_density_clustering.ipynb` — KMeans vs DBSCAN on noisy moon data, `eps` sweep, `min_samples` sweep, and a k-distance plot for picking `eps` systematically

## What I learned
- DBSCAN clusters by **density**: a point with at least `min_samples` neighbors within distance `eps` is a "core point"; clusters grow by chaining core points together. Points that don't belong to any dense region are labeled **noise (-1)** instead of being forced into a cluster
- This directly fixes KMeans's two weaknesses shown side by side: KMeans absorbed scattered outlier points into a cluster (it must assign every point somewhere), while DBSCAN correctly flagged them as noise
- DBSCAN also handles non-convex shapes (the moons) correctly, since it doesn't rely on centroids or straight-line boundaries — it just follows chains of nearby points
- `eps` is the most sensitive hyperparameter: too small and almost every point becomes noise (no neighbors within range); too large and separate clusters merge into one blob
- `min_samples` controls how "dense" a region must be to count as a cluster seed — lower values create more, smaller clusters; higher values are stricter and produce more noise
- The **k-distance plot** (sort each point's distance to its k-th nearest neighbor) is a practical heuristic for choosing `eps`: look for the "knee" where distance starts increasing sharply

## Concepts
- Density-based clustering
- Core points, noise points
- `eps` / `min_samples` hyperparameters
- k-distance plot heuristic

## Connections
- Direct comparison with: `19-kmeans-clustering/`
