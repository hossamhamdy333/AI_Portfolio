<div align="center">

# AI Portfolio

**Hossam Hamdy Fakry** — AI/ML Engineer & Data Analyst

Electronics and Communications Engineering graduate · ML/DL & deployment lead on a graduation Network Intrusion Detection System · Cairo, Egypt

[![GitHub](https://img.shields.io/badge/GitHub-hossamhamdy333-181717?logo=github&logoColor=white)](https://github.com/hossamhamdy333)
[![Email](https://img.shields.io/badge/Email-hossam3759180%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:hossam3759180@gmail.com)
[![Repo Size](https://img.shields.io/github/repo-size/hossamhamdy333/AI_Portfolio?color=blue)](https://github.com/hossamhamdy333/AI_Portfolio)
[![Last Commit](https://img.shields.io/github/last-commit/hossamhamdy333/AI_Portfolio?color=orange)](https://github.com/hossamhamdy333/AI_Portfolio)

`Python` `SQL` `PyTorch` `scikit-learn` `XGBoost` `LangChain` `LlamaIndex` `FastAPI` `MLflow` `DVC` `Docker` `Power BI`

</div>

---

Every project below goes from raw data to a served, evaluated result — with real numbers pulled from executed notebooks, not claimed in prose. Each project folder is self-contained: its own README, `requirements.txt`, and, where relevant, notebooks, SQL, a dashboard, and a live deployed demo.

### At a glance

| | |
|---|---|
| **Projects** | 17, spanning LLM/RAG, fine-tuning, classification, forecasting, and BI |
| **Live deployments** | 4 — DocuMind, Azure RAG Assistant, AI Support Copilot, arXiv Semantic Search |
| **Graduation project** | ML-NIDS — cascade intrusion detection on a 76M-row NetFlow dataset |
| **Core stack** | Python, SQL, PyTorch, scikit-learn/XGBoost/LightGBM, LangChain/LlamaIndex, FastAPI, MLflow, DVC |

### Contents

- [LLM / RAG / Fine-Tuning](#llm--rag--fine-tuning)
- [Data Analytics & Business Intelligence](#data-analytics--business-intelligence)
- [Classification / Regression](#classification--regression)
- [Graduation Project](#graduation-project)
- [Running any project](#running-any-project)
- [Contact](#contact)

---

## LLM / RAG / Fine-Tuning

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`allam_finetune`](./allam_finetune) | QLoRA fine-tuning of ALLaM-7B-Instruct on Arabic legal instruction data (article analysis, plain-language simplification, judgment prediction), LLM-judged against the zero-shot base model | `Python` `PyTorch` `Transformers` `PEFT/QLoRA` `bitsandbytes` | Faithfulness 4.72 → 7.63/10, relevance 7.15 → 8.77/10, fluency 8.94 → 9.11/10 — beats baseline on every metric and task type |
| [`nl2sql_finetune`](./nl2sql_finetune) | QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct for schema-constrained text-to-SQL generation | `Python` `PyTorch` `Transformers` `TRL` `PEFT/QLoRA` `SQLite` | Exact match 5.00% → 81.25%, valid SQL rate 92.00% → 97.25% |
| [`semantic-search-arxiv-papers`](./semantic-search-arxiv-papers) | Search engine over 50K arXiv ML abstracts, built up in stages (BM25 → SBERT+FAISS → Qdrant → cross-encoder reranking), each stage benchmarked on the same eval set | `Python` `rank-bm25` `Sentence-Transformers` `FAISS` `Qdrant` `FastAPI` `Streamlit` `DVC` | Reranking gives the largest jump: MRR 0.753 → 0.818, Recall@1 0.670 → 0.760 |
| [`rag_qa_documind`](./rag_qa_documind) | RAG Q&A over user-uploaded PDF/TXT/MD documents, per-session isolated vector store, answers grounded with cited source | `Python` `FastAPI` `ChromaDB` `Sentence-Transformers` `Gemini API` `Streamlit` | **Live:** [documents-mind.streamlit.app](https://documents-mind.streamlit.app/) |
| [`rag_router`](./rag_router) | Multi-domain RAG that routes a question to one of four topic indexes before answering; compares an LLM selector against an embedding selector | `Python` `LlamaIndex` `Qdrant Cloud` `Gemini API` `MLflow` `DVC` | LLM selector: routing accuracy 0.6975, MRR 0.6319 vs. embedding selector's 0.6450 / 0.6010 — but fails to parse 9.75% of the time vs. 1.0% |
| [`rag-vanilla-vs-langchain`](./rag-vanilla-vs-langchain) | Two RAG implementations over the same Arabic XLSum corpus and eval set — flat chunking vs. LangChain's ParentDocumentRetriever — isolating the effect of retrieval architecture | `Python` `LangChain` `Qdrant` `ChromaDB` `Gemini API` `MLflow` `DVC` `LangSmith` | LangChain wins on MRR (0.925 vs 0.802) and answer relevancy; flat chunking wins on citation accuracy and faithfulness — neither dominates |
| [`fact_check_crew`](./fact_check_crew) | Three CrewAI agents (Researcher, Writer, Critic) with a verify-and-revise loop, tested against a single-pass LLM baseline on the same retrieved passages | `Python` `CrewAI` `Qdrant` `MLflow` | Hallucination rate 0.12 (baseline) → 0.08 (crew) |
| [`llm_api_integration`](./llm_api_integration) | FastAPI service wrapping the Gemini API: streaming, tool calling, schema-validated structured output, retry/backoff, per-request token and cost tracking to MLflow | `Python` `FastAPI` `Pydantic` `google-generativeai` `MLflow` | 19 unit tests covering retries, schema validation, tool dispatch, and cost math |
| [`Azure_RAG_Assistant`](./Azure_RAG_Assistant) | Document upload and RAG chat assistant deployed on Azure, with blob storage archiving and a restricted-AST calculator tool | `Python` `FastAPI` `LangChain` `Gemini API` `Qdrant` `Azure Blob Storage` `Docker` | **Live:** [azure-rag-assistant...azurewebsites.net](https://azure-rag-assistant-b6hqawe7eef6euaf.francecentral-01.azurewebsites.net) |
| [`customer_support_copilot`](./customer_support_copilot) | Support chatbot on a QLoRA-fine-tuned Llama-3-8B, GGUF-quantized to run on CPU-only Azure Container Apps, grounded with RAG over a support knowledge base | `Python` `FastAPI` `llama-cpp-python` `Sentence-Transformers` `ChromaDB` `Docker` | Response time cut from timing out at 4 minutes to ~15-20 seconds after quantization; live demo deployed |

## Data Analytics & Business Intelligence

SQL pipelines, statistical testing, and dashboards built on top of a model's real output — not mockups.

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`employee-attrition`](./employee-attrition) | HR analytics: SQL data modeling, classification model comparison, Kaplan-Meier and Cox survival analysis for when people leave, and a cost-of-attrition model tied to dollar figures and retention ROI, with a Power BI dashboard | `Python` `SQL` `scikit-learn` `LightGBM` `lifelines` `Streamlit` `Power BI` | Best: LightGBM, PR-AUC 0.578; Cox model shows overtime workers leave at ~3.2x the rate (hazard ratio 3.19, p < 0.005); est. $10.15M annual attrition cost |
| [`ecommerce-demand-forecasting`](./ecommerce-demand-forecasting) | Daily, SKU-level demand forecasting on the Online Retail II dataset, from a PostgreSQL cleaning pipeline through to inventory reorder recommendations, with a Power BI dashboard | `Python` `SQL` `scikit-learn` `XGBoost` `LightGBM` `SHAP` `Power BI` | Final model: 86.8% WAPE, beating both a zero-predict baseline (100%) and seasonal-naive (126.8%) |
| [`customer_churn_prediction`](./customer_churn_prediction) | Telecom churn model taken past the notebook: an independent SQL layer reproducing the segmentation, a Streamlit app, and a Power BI dashboard, all built on the model's real scored output | `Python` `SQL` `scikit-learn` `XGBoost` `LightGBM` `MLflow` `Streamlit` `Power BI` | Best: isotonic-calibrated Random Forest, ROC-AUC 0.84, recall 78%, precision 54% |
| [`marketing-ab-testing`](./marketing-ab-testing) | A/B test analysis of a 588,101-user ad campaign dataset in SQL (DuckDB) and Python — two-proportion z-test, effect size, power analysis, and ROI, with an interactive dashboard | `Python` `pandas` `statsmodels` `DuckDB` `Streamlit` | Conversion lift +0.77pp (p = 1.7e-13) but ROI 0.39x — the campaign is statistically real but did not pay for itself |

## Classification / Regression

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`Credit_Fraud_Detection`](./Credit_Fraud_Detection) | Fraud detection on a highly imbalanced (0.17% fraud) credit card transaction dataset, comparing class weighting against SMOTE-based resampling, with business-cost threshold tuning | `Python 3.12` `scikit-learn 1.8` `XGBoost 3.2` `LightGBM` `imbalanced-learn` `MLflow` | Best: XGBoost with class weights, AUC-PR 0.8183; cost-tuned threshold catches 83% of fraud vs. 79% at default |
| [`House_Price_Prediction`](./House_Price_Prediction) | Regression on the Ames Housing dataset, EDA through feature engineering, model comparison, Optuna tuning, and a stacking ensemble | `Python 3.12` `scikit-learn 1.8` `XGBoost 3.2` `LightGBM` `Optuna` `SHAP` `MLflow` | Best: stacking ensemble, CV RMSE 0.1104 (~11% average error) |
| [`sentiment_forge`](./sentiment_forge) | 5-class sentiment classification (SST-5) benchmarked three ways on the identical test set: TF-IDF+LogReg, BiLSTM+GloVe, fine-tuned BERT; best model exported to ONNX and pushed to Hugging Face Hub | `Python` `scikit-learn` `PyTorch` `Transformers` `ONNX` `DVC` `MLflow` | Best: BERT-base, F1 (weighted) 0.514 vs. 0.417 (TF-IDF) and 0.414 (BiLSTM) |

## Graduation Project

**ML-Based Network Intrusion Detection System (ML-NIDS)** — ML/DL and deployment lead role.

A cascade detection architecture (binary → 21-class multiclass → attack-only stage) trained on the ~76M-row NF-UQ-NIDS-v2 NetFlow dataset with 13 custom engineered features.

| | |
|---|---|
| **Models compared** | XGBoost, CatBoost, TabNet, Residual MLP, FT-Transformer (implemented from scratch) |
| **Explainability** | SHAP |
| **Deployment** | Real-time detection dashboard, validated in a GNS3-emulated enterprise network |
| **Documentation** | Full technical thesis in IEEE-style formatting |

---

## Running any project

Each folder has its own `requirements.txt` and README with exact setup and run instructions. General pattern:

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio
cd AI_Portfolio/<project-folder>
pip install -r requirements.txt
```

Some projects require a PostgreSQL or DuckDB instance for their SQL layer, Kaggle/Colab GPU access for training notebooks, or API keys (Gemini, Qdrant Cloud, Hugging Face) — see the project's own README for what's needed.

---

## Contact

<div align="center">

**Hossam Hamdy Fakry** · AI/ML Engineer & Data Analyst · Cairo, Egypt

[![GitHub](https://img.shields.io/badge/GitHub-hossamhamdy333-181717?logo=github&logoColor=white)](https://github.com/hossamhamdy333)
[![Email](https://img.shields.io/badge/Email-hossam3759180%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:hossam3759180@gmail.com)

</div>
