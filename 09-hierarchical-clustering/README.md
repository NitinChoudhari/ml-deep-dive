# Hierarchical Clustering with SciPy

Practical application of agglomerative hierarchical clustering using SciPy to discover natural groupings in customer data. Demonstrates the full unsupervised learning workflow: feature scaling, distance computation, linkage methods, dendrogram interpretation, and cluster extraction.

## Why SciPy instead of from scratch

Same engineering judgment as decision trees: I understand hierarchical clustering deeply (the merging algorithm, linkage methods, distance metrics, dendrogram interpretation). Implementing the nested loops and dynamic data structures from scratch isn't where the highest learning value is right now.

My implementation-from-scratch focus is reserved for what builds directly toward neural networks. For exploratory unsupervised methods, using `scipy.cluster.hierarchy` is the right call — it's optimized, well-tested, and exactly what I'd use in a real project.

The goal here is to demonstrate practical clustering analysis: feature scaling, choosing the right linkage method, reading dendrograms, and extracting meaningful clusters from the tree.

## My first taste of unsupervised learning

All my previous work in this repo (linear regression through decision trees) was **supervised learning** — I had labeled data and trained models to predict those labels. Hierarchical clustering is my first **unsupervised learning** implementation:

- No target labels
- The goal is to discover hidden structure
- Success is measured by domain validation, not accuracy
- Different mindset entirely

This shift is important. Real-world data often comes unlabeled. Knowing how to extract meaningful patterns from unlabeled data is a distinct skill from classification/regression.

## What I built

- `hierarchical_clustering_customers.ipynb` — complete clustering pipeline including:
  - Synthetic customer dataset with 3 natural groups (designed for visual verification)
  - Pre-clustering visualization to confirm visual structure
  - Feature scaling (critical for distance-based methods)
  - Hierarchical clustering with 4 different linkage methods compared side-by-side
  - Dendrogram visualization and interpretation
  - Cluster extraction at different "cut heights"
  - Post-clustering visualization showing discovered groups

## What I learned

### Feature scaling is non-negotiable for distance-based methods

My customer data had `AnnualSpending` (15-90, range 75) and `VisitsPerMonth` (1-15, range 14). Without scaling, spending dominated every distance calculation by 5x. After min-max scaling to [0,1] for both features, distance computations became meaningful.

**This is true for ALL distance-based methods:** K-means, KNN, SVMs, any clustering algorithm. Always scale first.

### Linkage methods produce very different trees

I tested all 4 linkage methods on the same data:

- **Single linkage:** Creates long, chain-like clusters. Sensitive to noise. Sometimes useful for detecting non-spherical clusters.
- **Complete linkage:** Creates tight, compact clusters. More robust but can create artificially balanced groups.
- **Average linkage:** A middle ground between single and complete.
- **Ward's linkage:** Minimizes within-cluster variance. Tends to produce balanced, well-separated clusters. **The safest default and what I used as primary.**

The dendrograms looked noticeably different for the same data, especially in how they grouped the "boundary" customers. This taught me that **clustering results are sensitive to algorithmic choices**, not just data. There's no single "correct" clustering — different methods reveal different structures.

### The dendrogram is the killer feature

Unlike K-means which requires you to specify `k` upfront, hierarchical clustering produces a full tree. You can:

1. Look at the dendrogram
2. See where natural "gaps" exist (big vertical distances between merges)
3. Cut the tree at that level to extract clusters

For my customer data, cutting at 3 clusters cleanly separated the 3 groups I had designed into the data. The algorithm discovered the structure I intended without me telling it how many clusters to expect.

### When clusters are "right"

Unsupervised learning has no ground truth to validate against. I judged clustering quality by:

1. **Visual inspection** — do the clusters match what I see in the scatter plot?
2. **Domain interpretability** — do the clusters mean something meaningful in context (e.g., "premium customers" vs "occasional shoppers")?
3. **Internal consistency** — are points within a cluster genuinely similar?

This is fundamentally different from supervised learning where accuracy gives you a clean number. Clustering quality is a judgment call.

## When to use hierarchical clustering

### Good fit:
- Small to medium datasets (< 10,000 points) — O(n²) complexity becomes prohibitive on larger data
- Don't know the number of clusters upfront — the dendrogram lets you explore
- Want to understand cluster relationships (which clusters are most similar)
- Need a deterministic result (unlike K-means which depends on random initialization)
- Biology/phylogenetics — literally invented for evolutionary trees
- Hierarchical structure naturally exists in the data (e.g., taxonomy)

### Bad fit:
- Large datasets — too slow
- Need to assign new points to clusters at inference time — hierarchical clustering gives you a grouping of the training data, not a predictive model
- Real-time / streaming applications

## Hierarchical vs K-Means decision matrix

| Criterion | Hierarchical | K-Means |
|-----------|--------------|---------|
| Need to specify k upfront? | No | Yes |
| Speed on large data | Slow | Fast |
| Deterministic results | Yes | No (depends on init) |
| Cluster relationships visualizable | Yes (dendrogram) | No |
| Predicts new points | Hard | Easy |
| Handles non-spherical clusters | Yes (single linkage) | No |

**My rule of thumb:** Hierarchical for exploration and small datasets. K-Means for production and large datasets. Often I'd use hierarchical to discover the right number of clusters, then use K-Means for the actual production pipeline.

## Linkage methods cheat sheet

| Method | Best when... |
|--------|-------------|
| Ward | General default. Want balanced, well-separated clusters. |
| Complete | Want compact, tight clusters. Robust to noise. |
| Average | Want a balance between single and complete. |
| Single | Detecting non-spherical or elongated clusters. Risk of chaining. |

If unsure, use **Ward**. It's sklearn's default and works well in most cases.

## Concepts demonstrated

- Supervised vs unsupervised learning (paradigm shift)
- Distance metrics (Euclidean)
- Feature scaling for distance-based methods
- Agglomerative clustering algorithm
- Linkage methods (single, complete, average, Ward)
- Dendrogram interpretation
- Choosing cluster count via dendrogram cuts
- Unsupervised learning evaluation (visual + domain knowledge)
- When to use which clustering algorithm

## Connections

- **Built on:** `04-multivariate-linear-regression/` (feature scaling intuition transfers)
- **Different paradigm from:** Everything else in this repo (first unsupervised method)
- **Leads to:** K-Means, DBSCAN, Gaussian Mixture Models, and other clustering methods