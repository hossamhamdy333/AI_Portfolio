# AI Portfolio

Hossam Hamdy Fakry — AI/ML Engineer & Data Analyst, Cairo, Egypt.

Electronics and Communications Engineering graduate, ML/DL and deployment lead on a graduation network intrusion detection system. This repo is a collection of applied ML and LLM projects: each one goes from raw data to a served, evaluated result, with real numbers pulled from executed notebooks rather than claimed in prose.

GitHub: https://github.com/hossamhamdy333
Email: hossam3759180@gmail.com

## How this repo is organized

Every project folder is self-contained with its own README, requirements, and (where relevant) notebooks, SQL, dashboards, and a deployed demo. The table below is a map, not a substitute — open a project's README for the full writeup, methodology, and how to run it.

## LLM / RAG / Fine-Tuning

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`allam_finetune`](./allam_finetune) | QLoRA fine-tuning of ALLaM-7B-Instruct on Arabic legal instruction data (article analysis, plain-language simplification, judgment prediction), LLM-judged against the zero-shot base model | Python, PyTorch, Transformers, PEFT/QLoRA, bitsandbytes | Faithfulness 4.72 to 7.63/10, relevance 7.15 to 8.77/10, fluency 8.94 to 9.11/10 — beats baseline on every metric and task type |
| [`nl2sql_finetune`](./nl2sql_finetune) | QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct for schema-constrained text-to-SQL generation | Python, PyTorch, Transformers, TRL (SFTTrainer), PEFT/QLoRA, SQLite | Exact match 5.00% to 81.25%, valid SQL rate 92.00% to 97.25% |
| [`semantic-search-arxiv-papers`](./semantic-search-arxiv-papers) | Search engine over 50K arXiv ML abstracts, built up in stages (BM25 to SBERT+FAISS to Qdrant to cross-encoder reranking), each stage benchmarked on the same eval set | Python, rank-bm25, Sentence-Transformers, FAISS, Qdrant, cross-encoder, FastAPI, Streamlit, DVC | Reranking gives the largest jump: MRR 0.753 to 0.818, Recall@1 0.670 to 0.760 |
| [`rag_qa_documind`](./rag_qa_documind) | RAG Q&A over user-uploaded PDF/TXT/MD documents, per-session isolated vector store, answers grounded with cited source | Python, FastAPI, ChromaDB, Sentence-Transformers, Gemini API, Streamlit | Deployed: https://documents-mind.streamlit.app/ |
| [`rag_router`](./rag_router) | Multi-domain RAG that routes a question to one of four topic indexes before answering; compares an LLM selector against an embedding selector | Python, LlamaIndex, Qdrant Cloud, Gemini API, MLflow (DagsHub), DVC | LLM selector: routing accuracy 0.6975, MRR 0.6319 vs. embedding selector's 0.6450 / 0.6010 — but fails to parse 9.75% of the time vs. 1.0% |
| [`rag-vanilla-vs-langchain`](./rag-vanilla-vs-langchain) | Two RAG implementations over the same Arabic XLSum corpus and eval set — flat chunking vs. LangChain's ParentDocumentRetriever — isolating the effect of retrieval architecture | Python, LangChain, Qdrant, ChromaDB, Gemini API, MLflow (DagsHub), DVC, LangSmith | LangChain wins on MRR (0.925 vs 0.802) and answer relevancy; flat chunking wins on citation accuracy and faithfulness — neither dominates |
| [`fact_check_crew`](./fact_check_crew) | Three CrewAI agents (Researcher, Writer, Critic) with a verify-and-revise loop, tested against a single-pass LLM baseline on the same retrieved passages | Python, CrewAI, Qdrant, MLflow (DagsHub) | Hallucination rate 0.12 (baseline) to 0.08 (crew) |
| [`llm_api_integration`](./llm_api_integration) | FastAPI service wrapping the Gemini API: streaming, tool calling, schema-validated structured output, retry/backoff, per-request token and cost tracking to MLflow | Python 3.12, FastAPI, Pydantic, google-generativeai, MLflow | 19 unit tests covering retries, schema validation, tool dispatch, and cost math |
| [`Azure_RAG_Assistant`](./Azure_RAG_Assistant) | Document upload and RAG chat assistant deployed on Azure, with blob storage archiving and a restricted-AST calculator tool | Python, FastAPI, LangChain, Gemini API, Qdrant, Hugging Face embeddings, Azure Blob Storage, Docker | Live: https://azure-rag-assistant-b6hqawe7eef6euaf.francecentral-01.azurewebsites.net |
| [`customer_support_copilot`](./customer_support_copilot) | Support chatbot on a QLoRA-fine-tuned Llama-3-8B, GGUF-quantized to run on CPU-only Azure Container Apps, grounded with RAG over a support knowledge base | Python, FastAPI, llama-cpp-python (GGUF), Sentence-Transformers, ChromaDB, Docker | Response time cut from timing out at 4 minutes to about 15-20 seconds after quantization; live demo deployed |

## Data Analytics & Business Intelligence

SQL pipelines, statistical testing, and dashboards built on top of a model's real output — not mockups.

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`employee-attrition`](./employee-attrition) | HR analytics: SQL data modeling, classification model comparison, Kaplan-Meier and Cox survival analysis for when people leave, and a cost-of-attrition model tied to dollar figures and retention ROI, with a Power BI dashboard | Python, SQL (PostgreSQL), scikit-learn, LightGBM, lifelines, Streamlit, Power BI | Best: LightGBM, PR-AUC 0.578; Cox model shows overtime workers leave at about 3.2x the rate (hazard ratio 3.19, p < 0.005); est. $10.15M annual attrition cost |
| [`ecommerce-demand-forecasting`](./ecommerce-demand-forecasting) | Daily, SKU-level demand forecasting on the Online Retail II dataset, from a PostgreSQL cleaning pipeline through to inventory reorder recommendations, with a Power BI dashboard | Python, SQL (PostgreSQL), scikit-learn, XGBoost, LightGBM, SHAP, Power BI | Final model: 86.8% WAPE, beating both a zero-predict baseline (100%) and seasonal-naive (126.8%) |
| [`customer_churn_prediction`](./customer_churn_prediction) | Telecom churn model taken past the notebook: an independent SQL layer reproducing the segmentation, a Streamlit app, and a Power BI dashboard, all built on the model's real scored output | Python, SQL (PostgreSQL), scikit-learn, XGBoost, LightGBM, SHAP, MLflow, Streamlit, Power BI | Best: isotonic-calibrated Random Forest, ROC-AUC 0.84, recall 78%, precision 54% |
| [`marketing-ab-testing`](./marketing-ab-testing) | A/B test analysis of a 588,101-user ad campaign dataset in SQL (DuckDB) and Python — two-proportion z-test, effect size, power analysis, and ROI, with an interactive dashboard | Python, pandas, statsmodels, DuckDB, Streamlit | Conversion lift +0.77pp (p = 1.7e-13) but ROI 0.39x — the campaign is statistically real but did not pay for itself |

## Classification / Regression

| Project | What it is | Stack | Key result |
|---|---|---|---|
| [`Credit_Fraud_Detection`](./Credit_Fraud_Detection) | Fraud detection on a highly imbalanced (0.17% fraud) credit card transaction dataset, comparing class weighting against SMOTE-based resampling, with business-cost threshold tuning | Python 3.12, scikit-learn 1.8, XGBoost 3.2, LightGBM, imbalanced-learn, MLflow 3.10 | Best: XGBoost with class weights, AUC-PR 0.8183; cost-tuned threshold catches 83% of fraud vs. 79% at default |
| [`House_Price_Prediction`](./House_Price_Prediction) | Regression on the Ames Housing dataset, EDA through feature engineering, model comparison, Optuna tuning, and a stacking ensemble | Python 3.12, scikit-learn 1.8, XGBoost 3.2, LightGBM, Optuna, SHAP, MLflow 3.10 | Best: stacking ensemble, CV RMSE 0.1104 (about 11% average error) |
| [`sentiment_forge`](./sentiment_forge) | 5-class sentiment classification (SST-5) benchmarked three ways on the identical test set: TF-IDF+LogReg, BiLSTM+GloVe, fine-tuned BERT; best model exported to ONNX and pushed to Hugging Face Hub | Python, scikit-learn, PyTorch, Transformers, ONNX, DVC, MLflow, GitHub Actions | Best: BERT-base, F1 (weighted) 0.514 vs. 0.417 (TF-IDF) and 0.414 (BiLSTM) |

## Graduation Project

**ML-Based Network Intrusion Detection System (ML-NIDS)** — ML/DL and deployment lead role. A cascade detection architecture (binary, then 21-class multiclass, then an attack-only stage) trained on the roughly 76M-row NF-UQ-NIDS-v2 NetFlow dataset with 13 custom engineered features. Models compared: XGBoost, CatBoost, TabNet, Residual MLP, and an FT-Transformer implemented from scratch, with SHAP explainability and a real-time detection dashboard validated in a GNS3-emulated enterprise network. The full technical thesis documents the architecture, experiments, and results in IEEE-style formatting.

## Running any project

Each folder has its own `requirements.txt` and README with exact setup and run instructions. General pattern:

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio
cd AI_Portfolio/<project-folder>
pip install -r requirements.txt
```

Some projects require a PostgreSQL or DuckDB instance for their SQL layer, Kaggle/Colab GPU access for training notebooks, or API keys (Gemini, Qdrant Cloud, Hugging Face) — see the project's own README for what's needed.

## Contact

Hossam Hamdy Fakry
AI/ML Engineer & Data Analyst, Cairo, Egypt
Email: hossam3759180@gmail.com
GitHub: https://github.com/hossamhamdy333
