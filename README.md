<div align="center">

# AI Portfolio

**Hossam Hamdy Fakry** | AI/ML Engineer & Data Analyst

Electronics and Communications Engineering graduate · ML/DL & deployment lead on a graduation Network Intrusion Detection System · Cairo, Egypt

[![GitHub](https://img.shields.io/badge/GitHub-hossamhamdy333-181717?logo=github&logoColor=white)](https://github.com/hossamhamdy333)
[![Email](https://img.shields.io/badge/Email-hossam3759180%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:hossam3759180@gmail.com)
[![Repo Size](https://img.shields.io/github/repo-size/hossamhamdy333/AI_Portfolio?color=blue)](https://github.com/hossamhamdy333/AI_Portfolio)
[![Last Commit](https://img.shields.io/github/last-commit/hossamhamdy333/AI_Portfolio?color=orange)](https://github.com/hossamhamdy333/AI_Portfolio)

`Python` `SQL` `PyTorch` `scikit-learn` `XGBoost` `LangChain` `LlamaIndex` `FastAPI` `MLflow` `DVC` `Docker` `Power BI`

</div>

---

Each project below has its own README covering the problem, approach, data, results and limitations. Numbers come from executed notebooks and committed result files. Where a headline number needed a caveat, the caveat is in the table. Each project folder is self-contained: its own README, requirements (`requirements.txt`, under `backend/` or `impl_*/` for the two multi-part projects) and, where relevant, notebooks, SQL, a dashboard, and a live deployed demo.

### At a glance

| | |
|---|---|
| **Projects** | 18, spanning LLM/RAG, agents, fine-tuning, classification, forecasting, and BI |
| **Live deployments** | 5: DocuMind, Azure RAG Assistant, AI Support Copilot, arXiv Semantic Search, Codebase Insight Agent |
| **Graduation project** | ML-NIDS: two-stage intrusion detection trained on a 22.8M-flow sample of the 76M-record NF-UQ-NIDS-v2 dataset |
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
| [`allam_finetune`](./allam_finetune) | QLoRA fine-tuning of ALLaM-7B-Instruct on Arabic legal instruction data (article analysis, plain-language simplification, judgment prediction), LLM-judged against the zero-shot base model | `Python` `PyTorch` `Transformers` `PEFT/QLoRA` `bitsandbytes` | Gemini-judged on 150 held-out rows: faithfulness 4.72 → 8.47/10, relevance 7.15 → 9.44/10, fluency 8.94 → 9.88/10 (v2 adapter). The first fine-tune scored below the base model until the LoRA scope, class balance and evaluation script were fixed |
| [`nl2sql_finetune`](./nl2sql_finetune) | QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct for schema-constrained text-to-SQL generation | `Python` `PyTorch` `Transformers` `TRL` `PEFT/QLoRA` `SQLite` | Exact match 5.00% → 81.25% as scored, or about 43% → 81.5% once single vs. double quote differences are ignored. Valid SQL rate 92.00% → 97.25% |
| [`semantic-search-arxiv-papers`](./semantic-search-arxiv-papers) | Search engine over 50K arXiv ML abstracts, built up in stages (BM25 → SBERT+FAISS → Qdrant → cross-encoder reranking), each stage benchmarked on the same 200 title-as-query test set | `Python` `rank-bm25` `Sentence-Transformers` `FAISS` `Qdrant` `FastAPI` `Streamlit` `DVC` | MRR 0.712 (BM25) → 0.753 (dense) → 0.818 (reranked). Recall@1 0.635 → 0.670 → 0.760. Reranking gives the largest jump. **Live:** [semantic-search-arxiv-papers.streamlit.app](https://semantic-search-arxiv-papers.streamlit.app) |
| [`rag_qa_documind`](./rag_qa_documind) | RAG Q&A over user-uploaded PDF/TXT/MD documents, with a separate vector collection per account and answers that cite their source chunk | `Python` `FastAPI` `ChromaDB` `Sentence-Transformers` `Gemini API` `Streamlit` | **Live:** [documents-mind.streamlit.app](https://documents-mind.streamlit.app/) · 30 tests pass; guardrails catch 19/20 (95%) on the shared adversarial set |
| [`rag_router`](./rag_router) | Multi-domain RAG that routes a question to one of four topic indexes before answering; compares an LLM selector against an embedding selector | `Python` `LlamaIndex` `Qdrant Cloud` `Gemini API` `MLflow` `DVC` | LLM selector: routing accuracy 0.6975, MRR 0.6319 vs. embedding selector's 0.6450 / 0.6010, but it fails to parse 9.75% of the time vs. 1.0% |
| [`rag-vanilla-vs-langchain`](./rag-vanilla-vs-langchain) | Two RAG implementations over the same Arabic XLSum corpus and eval set: flat chunking vs. LangChain's ParentDocumentRetriever | `Python` `LangChain` `Qdrant` `ChromaDB` `Gemini API` `MLflow` `DVC` `LangSmith` | LangChain wins on MRR (0.925 vs 0.802) and answer relevancy; flat chunking wins on citation accuracy and faithfulness. The two sides used different chunk-size units, so the gap is not a clean test |
| [`fact_check_crew`](./fact_check_crew) | Three CrewAI agents (Researcher, Writer, Critic) with a verify-and-revise loop, tested against a single-pass LLM baseline on the same retrieved passages | `Python` `CrewAI` `Qdrant` `MLflow` | Hallucination rate 0.12 (baseline) → 0.08 (crew) on 100 questions, at roughly 2-3x the LLM calls |
| [`llm_api_integration`](./llm_api_integration) | FastAPI service wrapping the Gemini API: streaming, JSON-routed tool calling, schema-validated structured output, retry/backoff, per-request token and cost tracking to MLflow, with Ollama and vLLM as switchable backends | `Python` `FastAPI` `Pydantic` `google-generativeai` `MLflow` | 27 unit tests covering retries, schema validation, tool dispatch, cost math and backend selection |
| [`Azure_RAG_Assistant`](./Azure_RAG_Assistant) | Document upload and RAG chat assistant deployed on Azure, with per-user retrieval, blob storage archiving and a restricted-AST calculator tool | `Python` `FastAPI` `LangChain` `Gemini API` `Qdrant` `Azure Blob Storage` `Docker` | **Live:** [azure-rag-assistant...azurewebsites.net](https://azure-rag-assistant-b6hqawe7eef6euaf.francecentral-01.azurewebsites.net) · 48 tests pass; guardrails catch 19/20 (95%) |
| [`customer_support_copilot`](./customer_support_copilot) | Support chatbot on a QLoRA-fine-tuned Llama-3-8B, GGUF-quantized to run on CPU-only Azure Container Apps, grounded with RAG over a support knowledge base | `Python` `FastAPI` `llama-cpp-python` `Sentence-Transformers` `ChromaDB` `Docker` | Response time cut from timing out to ~15-20 seconds after GGUF quantization and a thread-count fix. **Live:** [support-copilot-app...azurecontainerapps.io](https://support-copilot-app.blackpebble-352cd42a.francecentral.azurecontainerapps.io) |
| [`Codebase_Insight_Agent`](./Codebase_Insight_Agent) | LangGraph agent (plan → retrieve → critique → retry) that answers questions about this portfolio, grounded in each project's own README; served as an MCP server and a public no-login website | `Python` `LangGraph` `LlamaIndex` `Qdrant` `MCP` `FastAPI` `Azure Container Apps` | **Live:** [portfolio-agent...azurecontainerapps.io](https://portfolio-agent.livelystone-91518072.germanywestcentral.azurecontainerapps.io) · 41 tests pass; guardrails catch 19/20 (95%) on the shared adversarial set |

## Data Analytics & Business Intelligence

SQL pipelines, statistical testing, and dashboards built on top of a model's real output, not mockups.

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`employee-attrition`](./employee-attrition) | HR analytics: SQL data modeling, classification model comparison, Kaplan-Meier and Cox survival analysis for when people leave, and a cost-of-attrition model tied to dollar figures and retention ROI, with a Power BI dashboard | `Python` `SQL` `scikit-learn` `LightGBM` `lifelines` `Streamlit` `Power BI` | Best: LightGBM, 5-fold CV PR-AUC 0.578. Cox model shows overtime workers leave at ~3.2x the rate (hazard ratio 3.19, 95% CI 2.47 to 4.12). Estimated annual attrition cost of $10.15M under stated cost assumptions |
| [`ecommerce-demand-forecasting`](./ecommerce-demand-forecasting) | Daily, SKU-level demand forecasting on the Online Retail II dataset, from a PostgreSQL cleaning pipeline through to inventory reorder recommendations, with a Power BI dashboard | `Python` `SQL` `scikit-learn` `LightGBM` `SHAP` `Power BI` | Final model: 86.8% WAPE, beating both a zero-predict baseline (100%) and seasonal-naive (126.8%) |
| [`customer_churn_prediction`](./customer_churn_prediction) | Telecom churn model taken past the notebook: an independent SQL layer reproducing the segmentation, a Streamlit app, and a Power BI dashboard, all built on the model's real scored output | `Python` `SQL` `scikit-learn` `XGBoost` `LightGBM` `MLflow` `Streamlit` `Power BI` | Best: Random Forest (isotonic-calibrated for deployment), ROC-AUC 0.84; pre-calibration recall 78%, precision 54% |
| [`marketing-ab-testing`](./marketing-ab-testing) | A/B test analysis of a 588,101-user ad campaign dataset in SQL (DuckDB) and Python: two-proportion z-test, effect size, power analysis, and ROI, with an interactive dashboard | `Python` `pandas` `statsmodels` `DuckDB` `Streamlit` | Conversion lift +0.77pp (p = 1.7e-13) but ROI 0.39x. The campaign is statistically real but did not pay for itself |

## Classification / Regression

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`Credit_Fraud_Detection`](./Credit_Fraud_Detection) | Fraud detection on a highly imbalanced (0.17% fraud) credit card transaction dataset, comparing class weighting against SMOTE-based resampling, with business-cost threshold tuning | `Python 3.11` `scikit-learn 1.8` `XGBoost 3.2` `LightGBM` `imbalanced-learn` `MLflow` | Best: XGBoost with class weights, AUC-PR 0.8183 on a hold-out of 56,746 transactions (95 frauds). Cost-tuned threshold catches 83% of fraud vs. 79% at default |
| [`House_Price_Prediction`](./House_Price_Prediction) | Regression on the Ames Housing dataset, EDA through feature engineering, model comparison, Optuna tuning, and a stacking ensemble | `Python 3.11` `scikit-learn 1.8` `XGBoost 3.2` `LightGBM` `Optuna` `SHAP` `MLflow` | Best: stacking ensemble, 5-fold CV RMSE 0.1104 on log price (Ridge baseline 0.1119), about 11% typical error |
| [`sentiment_forge`](./sentiment_forge) | 5-class sentiment classification (SST-5) benchmarked three ways on the identical test set: TF-IDF+LogReg, BiLSTM+GloVe, fine-tuned BERT; best model exported to ONNX and pushed to Hugging Face Hub | `Python` `scikit-learn` `PyTorch` `Transformers` `ONNX` `DVC` `MLflow` | Best: BERT-base, weighted F1 0.51 vs. 0.42 (TF-IDF) and 0.41 (BiLSTM) |

## Graduation Project

**ML-Based Network Intrusion Detection System (ML-NIDS)**: ML/DL and deployment lead role.

Three XGBoost models were trained (binary, 21-class and attack-only) on a 22.8M-flow stratified sample of the ~76M-record NF-UQ-NIDS-v2 NetFlow dataset, using 13 engineered features. The deployed cascade has two stages: the binary model flags an attack, then the attack-only model names it.

| | |
|---|---|
| **Models compared** | XGBoost, CatBoost, TabNet, Residual MLP |
| **Best model** | XGBoost: 99.09% binary accuracy, 98.23% on 21 classes (0.82 macro-F1, pulled down by Analysis, Infiltration and MITM) |
| **Explainability** | XGBoost gain importance and TabNet attention masks |
| **Deployment** | Docker, FastAPI, NFStream packet capture and a live detection dashboard, tested on GNS3-emulated traffic |
| **Documentation** | Full written thesis |
| **Modeling repo** | [machine-learning-techniques-for-intrusion-detection](https://github.com/hossamhamdy333/machine-learning-techniques-for-intrusion-detection) |
| **Deployment repo** | [ids-deploy](https://github.com/hossamhamdy333/ids-deploy) |

---

## Running any project

Each folder has its own requirements file and README with exact setup and run instructions (`Azure_RAG_Assistant` keeps its file in `backend/`; `rag-vanilla-vs-langchain` has one per `impl_*/` folder). General pattern:

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio
cd AI_Portfolio/<project-folder>
pip install -r requirements.txt
```

Some projects require a PostgreSQL, DuckDB or Oracle XE instance for their SQL layer, Kaggle/Colab GPU access for training notebooks, or API keys (Gemini, Qdrant Cloud, Hugging Face). See the project's own README for what's needed.

---

## Contact

<div align="center">

**Hossam Hamdy Fakry** · AI/ML Engineer & Data Analyst · Cairo, Egypt

[![GitHub](https://img.shields.io/badge/GitHub-hossamhamdy333-181717?logo=github&logoColor=white)](https://github.com/hossamhamdy333)
[![Email](https://img.shields.io/badge/Email-hossam3759180%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:hossam3759180@gmail.com)

</div>
