# AI-ML-DeepDive

> My personal curriculum for going from GenAI integrator to ML engineer — math, classic ML, deep learning, NLP, generative AI, MLOps, and time series, in that order.

## Why this repo exists

I'm a Software Engineer at LTIMindtree with 3+ years of experience, currently shipping production GenAI systems (multi-agent LLM RAG, deployed across DEV/IMP/Production environments at a Fortune 500 client). I've worked extensively with LangChain, crewAI, OpenAI APIs, and vector databases.

But I could *use* ML libraries without being able to *build* the algorithms underneath. This repo is me closing that gap — working through math, classic ML, and eventually deep learning/NLP/GenAI/MLOps/time series from first principles, one numbered folder at a time.

## Folder structure

```
AI-ML-DeepDive/
├── 01_Mathematics/                  Vectors, matrices, eigenvalues, derivatives, probability, gradient descent
├── 02_Python_For_ML/                NumPy, Pandas, Matplotlib, Seaborn, Feature Engineering
├── 03_Machine_Learning/             26 ML algorithms (from-scratch NumPy + scikit-learn) — see its own README
├── 04_Deep_Learning/                Perceptron → ANN/CNN/RNN/LSTM/GRU/Attention/Transformers (TensorFlow, PyTorch)
├── 05_NLP/                          Tokenization/TF-IDF/Word2Vec → BERT/GPT/T5, classification, NER, summarization
├── 06_Generative_AI/                LLM fundamentals, prompting, vector DBs (FAISS/ChromaDB), RAG, agents (LangChain/LangGraph/MCP), fine-tuning (LoRA/QLoRA), eval (RAGAS/TruLens)
├── 07_MLOps/                        FastAPI, Docker, MLflow, DVC, CI/CD, deployment, monitoring
├── 08_Time_Series/                  ARIMA, SARIMA, Prophet, LSTM
└── 10-neural-network-from-scratch/  Standalone NN-from-scratch notebook (currently at repo root, not nested under 04_Deep_Learning)
```

Each numbered top-level folder is one stage of the curriculum. Inside, work is broken into `NN-topic-name/` subfolders (the pattern used throughout `03_Machine_Learning/`), each holding one or more executed Jupyter notebooks and usually a `README.md` with what was built, what I learned, and the underlying concepts.

## My learning approach

For every topic, I follow a 4-level mastery framework:

1. **Math** — derive the equations on paper from first principles
2. **Code** — implement from scratch where it teaches something (NumPy, no shortcuts); use the standard library (scikit-learn, TensorFlow, PyTorch, etc.) once the fundamentals are proven
3. **Intuition** — explain it in plain English to a beginner
4. **Failure modes** — know when, why, and how it breaks

I move on only when all 4 are solid.

## Connect

- **GitHub:** [@NitinChoudhari](https://github.com/NitinChoudhari)
- **LinkedIn:** [@Linkedin](www.linkedin.com/in/nitin-choudhari)

---

*Last updated: June 2026*
