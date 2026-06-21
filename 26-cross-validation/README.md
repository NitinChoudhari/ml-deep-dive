# Cross Validation

Why a single train/test split is unreliable, and how KFold, StratifiedKFold, GridSearchCV, and RandomizedSearchCV give more trustworthy performance estimates and hyperparameter tuning, on the wine dataset.

## What I built
- `cross_validation.ipynb` — accuracy variance across 20 different random train/test splits, KFold vs StratifiedKFold (including a demonstration of plain KFold producing skewed class distributions per fold on sorted data), `GridSearchCV` over an SVM's `C`/`gamma`, and `RandomizedSearchCV` over a Random Forest's hyperparameters

## What I learned
- Running the **same model on the same data** with only the `train_test_split` random seed changed produced a real spread in test accuracy — a single split can make a model look better or worse than it actually is, purely by chance
- **KFold cross-validation** averages performance over K different splits, giving both a more reliable mean estimate and a standard deviation that quantifies how stable that estimate is
- **StratifiedKFold** preserves class proportions within every fold. Demonstrated the failure mode directly: an unshuffled `KFold` on class-sorted data produced folds with skewed or entirely missing classes — a real bug that's easy to introduce silently if the data happens to be grouped or sorted
- **GridSearchCV** exhaustively cross-validates every combination in a hyperparameter grid — thorough but the combination count grows multiplicatively with each added parameter
- **RandomizedSearchCV** samples a fixed number of random combinations from (potentially continuous) distributions instead of trying everything — found a comparable best score to GridSearchCV here while trying far fewer combinations, which matters once the hyperparameter space gets large
- Visualizing GridSearchCV's full results as a heatmap over two hyperparameters shows the accuracy landscape directly — useful for spotting whether the optimum is a sharp peak (sensitive) or a broad plateau (robust)

## Concepts
- KFold / StratifiedKFold
- Grid search vs randomized search
- Variance of performance estimates

## Connections
- Used implicitly throughout: every `GridSearchCV`/`cross_val_score` call in `10-elasticnet-regression/`, `13-svm/`, `15-extra-trees/`, etc. relies on the concepts built here
