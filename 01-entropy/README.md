# Entropy from Scratch

Implementation of Shannon entropy in NumPy, applied to the classic "predict farm chicken sex distribution" problem.

## What I built
- `entropy.ipynb` — computes entropy for various probability distributions

## What I learned
- Entropy measures uncertainty in a probability distribution
- Maximum entropy occurs at uniform distribution (50/50 split = 1.0 bit for binary)
- Zero entropy when one outcome is certain
- Always handle the `log(0)` edge case — filter zero probabilities before computing

## Why this matters
Entropy is the foundation of:
- Cross-entropy loss (used to train every classifier including LLMs)
- Information gain (used in decision tree splits)
- Perplexity (used to evaluate language models)

## Concepts
- Information theory
- Probability distributions
- Edge case handling