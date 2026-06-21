# KMeans Clustering

Centroid-based clustering on the same customer-segmentation data style as `09-hierarchical-clustering/`, plus a look at where KMeans structurally fails.

## What I built
- `kmeans_customer_segmentation.ipynb` — elbow method (inertia vs K), silhouette score vs K, final cluster + centroid visualization, and a moon-shaped dataset showing KMeans's convexity limitation

## What I learned
- KMeans alternates between assigning points to the nearest centroid and recomputing centroids as the mean of their assigned points, until convergence — it's an iterative optimization, not a one-shot computation like hierarchical clustering's merge tree
- **Elbow method**: inertia (within-cluster sum of squared distances) always decreases as K increases — look for the point where the rate of decrease sharply slows (the "elbow"), not just the lowest value
- **Silhouette score** is a more objective alternative to eyeballing an elbow — it directly measures how well-separated and cohesive clusters are, and gave a cleaner signal for choosing K than the elbow plot did here
- KMeans assumes clusters are **convex and roughly spherical** because it partitions space using straight-line (Voronoi) boundaries around centroids. On moon-shaped data it was structurally unable to follow the curved boundary, no matter how it was tuned
- This is the direct motivation for density-based methods — see `20-dbscan-clustering/`

## Concepts
- Centroid-based partitioning
- Elbow method and silhouette score for choosing K
- Convexity assumption / Voronoi boundaries

## Connections
- Same dataset style as: `09-hierarchical-clustering/`
- Limitation addressed by: `20-dbscan-clustering/`
