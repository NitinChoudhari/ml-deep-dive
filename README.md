# ML Deep Dive

> From-scratch implementations of machine learning algorithms in pure NumPy. My journey from GenAI integrator to ML engineer.

## Why this repo exists

I'm a Software Engineer at LTIMindtree with 3+ years of experience, currently shipping production GenAI systems (multi-agent LLM RAG, deployed across DEV/IMP/Production environments at a Fortune 500 client). I've worked extensively with LangChain, crewAI, OpenAI APIs, and vector databases.

But I realized something: I could *use* ML libraries, but I couldn't *build* the algorithms underneath. Calling `model.fit()` is not the same as understanding gradient descent.

This repo is my answer. Every algorithm here is implemented from scratch in NumPy, with no high-level ML libraries during training. I derive the math on paper, code it from a blank file, and verify against sklearn.

## What's inside

| # | Topic | Key Concepts |
|---|-------|--------------|
| 01 | [Entropy](./01-entropy/) | Information theory, probability distributions |
| 02 | [Linear Regression (single feature)](./02-linear-regression-single-feature/) | Gradient descent, MSE loss, learning rate |
| 03 | [Linear Regression on real data](./03-linear-regression-real-data/) | Feature scaling, irreducible error |
| 04 | [Multivariate Linear Regression](./04-multivariate-linear-regression/) | Vectorized math, multi-feature gradient descent |
| 05 | [Cross-Entropy Loss](./05-cross-entropy/) | Surprise as loss, on/off switching, classification foundations |
| 06 | [Logistic Regression](./06-logistic-regression/) | Sigmoid, cross-entropy loss, decision boundary, confusion matrix |

More coming as I progress: logistic regression, decision trees, neural networks from scratch, CNNs, transformers, fine-tuning.

## My learning approach

For every algorithm, I follow a 4-level mastery framework:

1. **Math** — derive the equations on paper from first principles
2. **Code** — implement from scratch in NumPy, no shortcuts
3. **Intuition** — explain it in plain English to a beginner
4. **Failure modes** — know when, why, and how it breaks

I move on only when all 4 are solid.

## Stack

- Python 3.11
- NumPy (the only library used during training)
- Matplotlib (visualization)
- scikit-learn (sanity check only — never used during training)
- Jupyter Notebook

## Resources I'm learning from

- StatQuest with Josh Starmer — for conceptual depth
- Andrej Karpathy's Neural Networks: Zero to Hero — for implementation depth
- Original papers (when applicable)

## Connect

- **GitHub:** [@NitinChoudhari](https://github.com/NitinChoudhari)
- **LinkedIn:** [@Linkedin](www.linkedin.com/in/nitin-choudhari)
- **Production GenAI work:** see my [LogMind](https://github.com/YOUR_USERNAME/LogMind) repo

---

*Last updated: May 2026*
