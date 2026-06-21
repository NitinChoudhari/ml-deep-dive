# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML learning journal, not a software project. Each numbered folder is one topic, implemented as a Jupyter notebook with output cells baked in (executed, not just source). There is no application code, no package to install, no test suite, and no build step — "correctness" means a notebook runs top-to-bottom without errors and produces sensible output.

This folder (`03_Machine_Learning`) is one section of a larger curriculum at `G:\Learning\AI-ML-DeepDive\` (siblings: `01_Mathematics`, `02_Python_For_ML`, `04_Deep_Learning`, `05_NLP`, `06_Generative_AI`, `07_MLOps`, `08_Time_Series` — currently just placeholder `Topics to Cover.txt` files, no content yet).

## Roadmap and structure

`Topics to Cover.txt` in this folder is the checklist of what should eventually exist here, grouped by category (Regression, Classification, Tree Models, Clustering, Dimensionality Reduction, Model Evaluation). Cross-reference it against the numbered folders before assuming a topic is "missing" — check the folder list, not just the filename pattern.

Folders are numbered in the order they were built (`01-entropy` ... `26-cross-validation`), not strictly grouped by the categories in the topics file. Folder naming is `NN-topic-name` (kebab-case, lowercase). Each folder contains:
- One or more `.ipynb` notebooks (already executed — outputs/plots are part of the file)
- A `README.md` (not all folders have one yet — `06-logistic-regression`, `07-ridge-lasso-regression`, `08-decision-trees-sklearn`, and `09-hierarchical-clustering` currently lack one; this is a pre-existing gap, not intentional)

## Notebook conventions

- **Foundational/from-scratch topics** (entropy, linear regression, ridge/lasso, cross-entropy) are implemented in raw NumPy — manual gradient descent loops, no sklearn during training, sometimes a sklearn comparison at the end as a sanity check.
- **Practical/library-driven topics** (decision trees, hierarchical clustering, and everything added from `10-elasticnet-regression` onward — KNN, Naive Bayes, SVM, ensemble/boosting methods, clustering, dimensionality reduction, model evaluation) use scikit-learn (or xgboost/lightgbm/catboost/umap-learn) directly with a full pipeline: load data → explore/visualize → train/test split → train → evaluate (confusion matrix, classification report, or regression metrics) → visualize/interpret → tune hyperparameters.
- Toy datasets are often invented inline (not loaded from files) and lean on a recurring "Indian housing/customer" theme with ₹ currency where a regression/business framing fits. Classification/clustering/dimensionality-reduction notebooks mostly use scikit-learn's bundled datasets (`load_wine`, `load_breast_cancer`, `load_digits`, `load_diabetes`) so nothing needs downloading.
- Notebooks favor grouping related code (e.g. preprocessing + training) into a single cell after `04-multivariate-linear-regression` hit a cell-execution-order bug from running training with stale variables — re-running cells out of order is a real failure mode here, not just a style preference.

## README.md convention

When a folder has a README, it follows this shape — match it for any new topic:
```
# Title

One-line description of what's implemented and on what data.

## What I built
- bullet list of notebook(s) and what each does

## What I learned
- first-person, specific insights — not generic textbook definitions

## Math (optional, when there's a formula worth pinning down)

## Concepts
- short tag list of the underlying ideas

## Connections (optional)
- links to related folders, e.g. "Builds on: `07-ridge-lasso-regression/`"
```

## Environment

- Python 3.13 (`py -3` on this Windows machine — plain `python` is not on PATH, it resolves to the Microsoft Store shim)
- Core packages: numpy, scipy, matplotlib, pandas, seaborn, scikit-learn
- Extra packages used by specific topics: xgboost, lightgbm, catboost, umap-learn — only imported in the notebooks that need them, not a blanket dependency
- A Jupyter kernel named `python3` is registered for this interpreter (`py -3 -m ipykernel install --user --name python3`) — needed for any notebook-execution tooling (`nbclient`, `jupyter nbconvert --execute`) to find the right environment
- No `requirements.txt`/`environment.yml` exists — if one is added later, keep heavy boosting/UMAP libraries optional or clearly separated, since not every topic needs them

## Working in this repo

- To verify a notebook is still correct, execute it end-to-end and check for error outputs rather than just reading the source — a notebook with no output cells, or with stale output from a previous version of the code, is not verified.
- Don't add a `.gitignore`-worthy `.ipynb_checkpoints/` folder's contents to a commit; several already exist under `01-entropy`, `03-linear-regression-real-data`, `06-logistic-regression`, `07-ridge-lasso-regression`, `08-decision-trees-sklearn`, `09-hierarchical-clustering`.
