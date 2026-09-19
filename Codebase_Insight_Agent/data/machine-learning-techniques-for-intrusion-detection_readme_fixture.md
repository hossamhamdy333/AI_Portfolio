# Machine Learning Techniques for Intrusion Detection

A large-scale comparative study of modern machine learning and deep learning approaches for **network intrusion detection** using the **NF-UQ-NIDS-v2** benchmark dataset.

The project evaluates multiple architectures under the same preprocessing pipeline, train/validation/test splits, and evaluation metrics to provide a fair comparison between:

* Gradient boosted decision trees
* Attention-based tabular neural networks
* Deep residual multilayer perceptrons
* Classical statistical preprocessing pipelines

The repository is designed for reproducible experimentation on the **Kaggle free tier (T4 GPU)** while still scaling to tens of millions of network flow records.

---

## Project Goals

## This repository focuses on:


 Scalable IDS Pipelines: Building high-throughput data processing networks capable of parsing and normalizing massive NetFlow/IPFIX datasets without packet loss. 
 
 Dual-Architecture Benchmarking: Executing continuous, rigorous comparisons between traditional tree-based classifiers (`XGBoost`, `CatBoost`) and specialized tabular deep learning paradigms (`PyTorch`, `TabNet`).
 
 Imbalance Robustness: Evaluating, engineering, and stabilizing model loss gradients under the severe class imbalances (often >99.9% benign) inherent to real-world network traffic.
 
 Explainable AI (XAI): Transitioning from black-box heuristics to trusted automation by computing global and real-time local `SHAP` feature attribution vectors for security analysts.
 
 Research Verification: Benchmarking architecture performance, False Positive Rates (FPR), and Macro F1-Scores directly against published academic baselines on the **NF-UQ-NIDS-v2** multi-source dataset.
 
Real-World Deployment: Bridging the gap between Jupyter Notebook prototyping and enterprise infrastructure using distributed stream processing (`Kafka`/`Spark`), line-rate inference engines (`ONNX Runtime`), and structural `SIEM` alerting layers.

---

## Dataset

### NF-UQ-NIDS-v2

The experiments use the publicly available:

**Network Flow University of Queensland Intrusion Detection System Dataset v2 (NF-UQ-NIDS-v2)**

### Dataset Characteristics

| Property       | Value                         |
| -------------- | ----------------------------- |
| Records        | ~76 million                   |
| Features       | 42 numerical NetFlow features |
| Classes        | 21 total classes              |
| Benign classes | 1                             |
| Attack classes | 20                            |
| Format         | CSV                           |
| Feature type   | Continuous numerical          |
| Domain         | Network intrusion detection   |

### Source

* Kaggle: [https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset](https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset)
* Original dataset page: [https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

### Challenges of the Dataset

NF-UQ-NIDS-v2 is significantly harder than many small IDS benchmarks because it contains:

* Extremely imbalanced classes
* Rare attack categories with only hundreds of samples
* Large-scale traffic distributions
* Highly skewed NetFlow statistics
* Strong feature correlations
* Non-linear decision boundaries

Several attack classes contain fewer than 1,000 samples, making macro-level evaluation metrics critical.

---

## Repository Structure

```text
├── notebooks/
│   ├── eda/
│   ├── preprocessing/
│   ├── xgboost/
│   ├── catboost/
│   ├── tabnet/
│   └── residual-mlp/
│
├── plots/
│   ├── eda/
│   ├── preprocessing/
│   ├── xgboost/
│   ├── catboost/
│   ├── tabnet/
│   └── residual-mlp/
│
├── LICENSE
└── README.md
```

---

# Pipeline Overview

Each modelling notebook trains three IDS variants:

| Model   | Objective                      | Output            |
| ------- | ------------------------------ | ----------------- |
| Model 1 | Binary detection               | Benign vs Attack  |
| Model 2 | Full multiclass classification | 21 classes        |
| Model 3 | Attack-only classification     | 20 attack classes |

This design enables deployment as a two-stage IDS pipeline:

1. Detect malicious traffic
2. Identify the attack category

---

# Notebook Details

## 01 — Exploratory Data Analysis

**Directory:** `notebooks/eda/`

### Main Objectives

* Understand class imbalance
* Analyse feature distributions
* Detect multicollinearity
* Identify skewed variables
* Evaluate feature separability
* Discover informative features

### Key Techniques

* Chunked CSV loading
* Stratified sampling
* Correlation analysis
* Mutual information ranking
* PCA visualisation
* Skewness and kurtosis analysis
* Duplicate detection
* Missing value analysis

### Important Findings

* The dataset is highly imbalanced
* Three dominant classes represent most traffic
* Multiple features are strongly skewed
* PCA shows weak linear separability
* Several highly correlated features exist
* Engineered ratio features improve separability

### Outputs

* Correlation heatmaps
* Distribution plots
* PCA projections
* Mutual information rankings
* Class imbalance charts

---

## 02 — Preprocessing & Feature Engineering

**Directory:** `notebooks/preprocessing/`

### Main Objectives

* Prepare data for all downstream models
* Create reusable preprocessing artefacts
* Reduce redundancy
* Improve feature quality
* Handle imbalance safely

### Preprocessing Pipeline

| Step                | Description                |
| ------------------- | -------------------------- |
| Missing handling    | Median imputation          |
| Scaling             | RobustScaler               |
| Encoding            | LabelEncoder               |
| Feature selection   | Mutual information         |
| Correlation pruning | Remove redundant variables |
| Transformations     | log1p for skewed variables |
| Splitting           | Stratified train/val/test  |

### Engineered Features

The notebook introduces domain-informed networking features such as:

* Packet asymmetry
* Byte asymmetry
* Upload ratio
* Protocol flags
* SYN-only detection
* Retransmission ratio
* TTL spread
* Flow duration metrics

### Saved Artefacts

The notebook exports:

* NumPy arrays
* Label encoders
* Class weights
* Scalers
* Imputers
* Feature rankings
* Attack-only splits

These artefacts are reused by every model notebook.

---

## 03 — XGBoost

**Directory:** `notebooks/xgboost/`

### Overview

XGBoost serves as the primary tree-based baseline using GPU acceleration.

### Why XGBoost?

* Extremely fast training
* Strong tabular performance
* Stable optimisation
* Good handling of heterogeneous distributions
* High interpretability via gain importance

### Training Characteristics

| Property                 | Value               |
| ------------------------ | ------------------- |
| Backend                  | GPU histogram trees |
| Early stopping           | Yes                 |
| Feature scaling required | No                  |
| Interpretability         | High                |
| Memory efficiency        | Strong              |

### Strengths Observed

* Excellent binary detection performance
* Fast convergence
* Strong robustness to noisy features
* Reliable performance on dominant attack classes

### Weaknesses Observed

* Lower recall on ultra-rare attacks
* Less expressive than attention-based networks
* Feature interactions are implicit rather than explicit

---

## 04 — CatBoost

**Directory:** `notebooks/catboost/`

### Overview

CatBoost uses ordered boosting and symmetric decision trees to reduce overfitting and improve stability on imbalanced datasets.

### Why CatBoost?

* Better handling of imbalance
* Strong regularisation
* Robust validation behaviour
* Reduced target leakage
* Stable multiclass optimisation

### Training Characteristics

| Property             | Value            |
| -------------------- | ---------------- |
| Backend              | GPU              |
| Tree type            | Symmetric trees  |
| Overfitting detector | Native           |
| Validation strategy  | Ordered boosting |
| Interpretability     | High             |

### Strengths Observed

* Strong multiclass performance
* Better rare-class stability than XGBoost
* Smooth optimisation curves
* Consistent macro-F1 behaviour

### Weaknesses Observed

* Slightly slower training than XGBoost
* Higher memory usage during training

---

## 05 — TabNet

**Directory:** `notebooks/tabnet/`

### Overview

TabNet is an attention-based architecture designed specifically for tabular data.

Instead of using all features equally, TabNet learns sparse feature selection masks at each decision step.

### Why TabNet?

* Interpretable attention masks
* Dynamic feature selection
* Better modelling of complex feature interactions
* Neural network flexibility for tabular data

### Architecture Characteristics

| Component                | Value |
| ------------------------ | ----- |
| Decision dimension       | 16    |
| Attention dimension      | 16    |
| Decision steps           | 3     |
| Sparse attention         | Yes   |
| Mixed precision training | Yes   |

### Unique Contributions

The notebook includes:

* Per-class attention heatmaps
* Mean feature attention analysis
* Sparse feature selection visualisations
* Attack-specific feature focus analysis

### Strengths Observed

* Excellent interpretability
* Better representation learning for rare attacks
* Learns attack-specific feature subsets

### Weaknesses Observed

* Longest training time
* More sensitive to hyperparameters
* Higher GPU memory consumption

---

## 06 — Residual MLP

**Directory:** `notebooks/residual-mlp/`

### Overview

A deep residual multilayer perceptron implemented in PyTorch using:

* Residual connections
* Batch normalisation
* GELU activations
* Mixed precision training
* AdamW optimisation

### Why Residual MLP?

The architecture provides a simpler and faster alternative to Transformer-style tabular networks while maintaining strong predictive performance.

### Architecture Characteristics

| Component        | Value |
| ---------------- | ----- |
| Hidden dimension | 512   |
| Residual blocks  | 4     |
| Activation       | GELU  |
| Dropout          | 0.15  |
| Optimiser        | AdamW |
| Batch size       | 8192  |

### Strengths Observed

* Fast neural-network training
* Stable optimisation
* Strong macro-F1 performance
* Better scalability than attention-heavy architectures

### Weaknesses Observed

* Less interpretable than TabNet
* Gradient importance is noisier than tree importance

---

# Model Performance Comparison

## Measured Performance Results

> The following results are taken directly from the executed notebook outputs on the NF-UQ-NIDS-v2 dataset using the shared preprocessing pipeline.

## Model 1 — Binary Classification (Benign vs Attack)

| Model        | Accuracy | F1-Score | Weighted-F1 |
| ------------ | -------- | -------- | ----------- |
| XGBoost      | 0.9909   | 0.9897   | 0.9909      |
| CatBoost     | 0.9901   | 0.9888   | 0.9901      |
| TabNet       | 0.9831   | 0.9809   | 0.9831      |
| Residual MLP | 0.9813   | 0.9786   | 0.9812      |

### Binary Classification Observations

* All models exceed 98% accuracy
* XGBoost achieves the strongest overall binary detection performance
* Tree-based models converge faster and generalise better on dominant traffic distributions
* Neural architectures remain competitive while requiring significantly longer training

---

## Model 2 — Full Multiclass Classification (21 Classes)

| Model        | Accuracy | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1 |
| ------------ | -------- | ---------------------- | ------------------- | --------------- |
| XGBoost      | 0.9823   | 0.9825                 | 0.9823              | 0.9820          |
| CatBoost     | 0.9770   | 0.9774                 | 0.9770              | 0.9762          |
| TabNet       | 0.9615   | 0.9633                 | 0.9615              | 0.9627          |
| Residual MLP | 0.9659   | 0.9675                 | 0.9659              | 0.9669          |

### Full Multiclass Observations

The large gap between Weighted-F1 and F1-Score reflects the extreme imbalance of NF-UQ-NIDS-v2.

Dominant classes:

* DDoS
* DoS
* scanning
* xss
* Reconnaissance

contain hundreds of thousands to millions of samples.

Rare classes such as:

* Worms
* Shellcode
* Analysis
* Theft

contain fewer than 200 samples and heavily reduce F1-Score.

XGBoost provides the best overall multiclass performance and strongest rare-class stability among all evaluated models.

---

## Model 3 — Attack-Only Multiclass Classification (20 Attack Classes)

| Model        | Accuracy | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1 |
| ------------ | -------- | ---------------------- | ------------------- | --------------- |
| XGBoost      | 0.9872   | 0.9873                 | 0.9872              | 0.9872          |
| CatBoost     | 0.9816   | 0.9819                 | 0.9816              | 0.9815          |
| Residual MLP | 0.9791   | 0.9793                 | 0.9791              | 0.9792          |
| TabNet       | 0.9733   | 0.9742                 | 0.9733              | 0.9739          |

### Attack-Only Classification Observations

Removing benign traffic significantly improves F1-Score across all architectures.

This indicates that:

* benign-vs-attack separation is relatively easy
* distinguishing between attack families is substantially harder
* rare attack categories dominate F1-Score behaviour

The attack-only setup provides a more realistic benchmark for advanced IDS research.

---

## Rare-Class Behaviour

The following classes remain consistently difficult across all models:

| Class     | Main Challenge                     |
| --------- | ---------------------------------- |
| Worms     | Extremely low support (21 samples) |
| Analysis  | Severe overlap with other traffic  |
| Theft     | Very small support                 |
| mitm      | Weak statistical separability      |
| Shellcode | Small sample size                  |

Examples observed directly from notebook outputs:

* TabNet failed to correctly identify Worms samples in several runs
* Residual MLP struggled with Analysis precision despite strong recall
* CatBoost improved stability on Worms and Shellcode
* XGBoost produced the strongest overall F1-Score values

---

## Training & Runtime Comparison

| Model        | Approx Runtime | GPU Usage | Notes                  |
| ------------ | -------------- | --------- | ---------------------- |
| XGBoost      | ~120 min        | Moderate  | Fastest convergence    |
| CatBoost     | ~50 min        | Moderate  | Most stable training   |
| TabNet       | ~100 min        | High      | Most expensive model   |
| Residual MLP | ~40 min        | Moderate  | Best neural efficiency |

---

### Performance Summary

| Model        | Binary Detection | Full Multiclass | Attack-Only | Overall Trend              |
| ------------ | ---------------- | --------------- | ----------- | -------------------------- |
| XGBoost      | Best             | Best            | Best        | Strongest overall baseline |
| CatBoost     | Very Strong      | Strong          | Strong      | Stable under imbalance     |
| TabNet       | Strong           | Moderate        | Strong      | Best interpretability      |
| Residual MLP | Strong           | Strong          | Strong      | Best neural efficiency     |

## Overall Behaviour

The experiments show clear differences between classical boosting methods and deep learning approaches.

| Model        | Training Speed | Inference Speed | Rare Attack Performance | Interpretability | Memory Usage |
| ------------ | -------------- | --------------- | ----------------------- | ---------------- | ------------ |
| XGBoost      | Very Fast      | Very Fast       | Moderate                | High             | Moderate     |
| CatBoost     | Fast           | Fast            | Strong                  | High             | Moderate     |
| TabNet       | Slow           | Moderate        | Strong                  | Very High        | High         |
| Residual MLP | Fast           | Very Fast       | Strong                  | Moderate         | Moderate     |

---

## Binary Detection Performance

### General Findings

All models achieve very high binary detection capability because separating benign from malicious traffic is easier than distinguishing attack categories.

Observed behaviour:

* Tree models converge fastest
* Neural networks require longer warmup
* CatBoost produces the most stable validation curves
* Residual MLP achieves strong throughput-performance balance

---

## Full Multiclass Performance

### Most Difficult Classes

The following attack types are consistently difficult across all models:

* Worms
* Shellcode
* Analysis
* Fuzzers
* DDoS-SlowRate

Main reasons:

* Very small sample counts
* Overlapping traffic signatures
* Similar flow statistics
* High variance distributions

### Most Separable Classes

The easiest classes include:

* DDoS
* DoS
* Reconnaissance
* Scanning

These attacks exhibit stronger statistical patterns in:

* Packet counts
* Throughput
* TTL statistics
* Directional asymmetry

---

## Rare-Class Behaviour

### Best Rare-Class Stability

| Model        | Rare-Class Behaviour           |
| ------------ | ------------------------------ |
| CatBoost     | Most stable                    |
| TabNet       | Strong representation learning |
| Residual MLP | Competitive with weighting     |
| XGBoost      | Sensitive to imbalance         |

The combination of:

* log-smoothed weights
* engineered features
* balanced sampling
* robust preprocessing

substantially improves macro-level performance.

---

# Evaluation Metrics

Every notebook evaluates models using:

| Metric            | Purpose                      |
| ----------------- | ---------------------------- |
| Accuracy          | Overall correctness          |
| Weighted Avg F1   | Class-balanced evaluation    |
| Balanced Accuracy | Imbalance-aware accuracy     |
| ROC-AUC           | Binary separability          |
| PR-AUC            | Precision-recall behaviour   |
| Precision         | False positive control       |
| Recall            | Attack detection sensitivity |
| Confusion Matrix  | Per-class analysis           |

---

# Visualisations Included

Each modelling notebook generates:

* Learning curves
* ROC curves
* Precision-recall curves
* Normalised confusion matrices
* Per-class F1 charts
* Precision/recall heatmaps
* Feature importance rankings
* Published-paper comparison charts

Additional model-specific visualisations:

| Model        | Unique Visualisation              |
| ------------ | --------------------------------- |
| TabNet       | Attention masks                   |
| Residual MLP | Gradient importance maps          |
| XGBoost      | Gain importance                   |
| CatBoost     | PredictionValuesChange importance |

---

# Comparison with Published Research

The repository includes comparisons against previously published work on NF-UQ-NIDS-v2.

Compared approaches include:

* Random Forest
* CNNs
* LSTMs
* Transformers
* GNNs
* Extra Trees
* Hybrid architectures

The goal is not only to maximise accuracy, but also to compare:

* scalability
* training efficiency
* interpretability
* robustness under imbalance
* deployment practicality

---

# Hardware & Runtime

All notebooks are designed for:

| Resource | Configuration |
| -------- | ------------- |
| Platform | Kaggle        |
| GPU      | NVIDIA T4     |
| VRAM     | 15 GB         |
| RAM      | 30 GB         |

### Approximate Runtime

| Notebook      | Runtime |
| ------------- | ------- |
| EDA           | ~20 min |
| Preprocessing | ~30 min |
| XGBoost       | ~120 min |
| CatBoost      | ~50 min |
| TabNet        | ~100 min |
| Residual MLP  | ~40 min |

---

# Dependencies

```text
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib
seaborn
joblib
psutil
xgboost
catboost
pytorch-tabnet
torch==2.3.0
```

---

# How to Run

1. Upload NF-UQ-NIDS-v2 to Kaggle
2. Enable GPU acceleration
3. Run notebooks in order:

```text
01 → EDA
02 → Preprocessing
03 → XGBoost
04 → CatBoost
05 → TabNet
06 → Residual MLP
```

4. Download generated plots from the corresponding `/plots/` directory.

---

# Key Takeaways

* Gradient boosting remains extremely competitive for IDS tasks
* CatBoost provides the best balance between stability and performance
* TabNet offers the best interpretability among neural models
* Residual MLP achieves strong performance with significantly lower complexity
* Proper preprocessing and imbalance handling matter as much as architecture choice
* Rare-class evaluation is essential for realistic IDS benchmarking

---

# Author
**Hossam Hamdy**

# Connect with Me
* **Gmail:** [hossam3759180@gmail.com](mailto:hossam3759180@gmail.com)



-e 

---

# Supplementary document: `plots/README.md`

# Plots

All evaluation visualisations produced by the model notebooks. Each subfolder corresponds to one notebook and is populated when that notebook is run on Kaggle.

## Contents per method

| Subfolder | Notebook |
|-----------|---------|
| `eda/` | Notebook 01 — EDA |
| `preprocessing/` | Notebook 02 — Preprocessing |
| `xgboost/` | Notebook 03 — XGBoost |
| `catboost/` | Notebook 04 — CatBoost |
| `tabnet/` | Notebook 05 — TabNet |
| `residual-mlp/` | Notebook 06 — Residual MLP |

## Plot types (model notebooks)

Each model notebook saves the following plots:

**Binary model (Model 1)**
- `bin_learning_curve.png` — train loss + val AUC per epoch/iteration
- `bin_confusion_matrix.png` — raw counts + normalised heatmap side by side
- `bin_roc_curve.png` — ROC curve with AUC score
- `bin_pr_curve.png` — Precision-Recall curve with AP score
- `bin_feature_importance.png` — top 20 features, engineered highlighted in red

**Full multiclass (Model 2)**
- `multi_learning_curve.png`
- `multi_confusion_matrix.png`
- `multi_f1_per_class.png` — horizontal bar chart sorted by F1
- `multi_pr_heatmap.png` — Precision/Recall heatmap across 21 classes
- `multi_feature_importance.png`

**Attack-only multiclass (Model 3)**
- `att_learning_curve.png`
- `att_confusion_matrix.png`
- `att_f1_per_class.png`
- `att_pr_heatmap.png`
- `att_feature_importance.png`

**Model-specific**
- `tabnet_attention_mean.png` — mean CLS/step attention bar chart
- `tabnet_attention_per_class.png` — per-class attention heatmap (TabNet)
- `resmlp_attention_mean.png` — mean gradient importance bar chart
- `resmlp_attention_per_class.png` — per-class gradient importance heatmap
- `papers_comparison.png` — your models vs published work bar chart

## Downloading from Kaggle

At the end of each notebook, all plots are zipped and available as a notebook output:

```python
shutil.make_archive('xgboost_plots', 'zip', plots_dir)
```

Download the zip from the Kaggle notebook output panel and extract into the corresponding subfolder here.
-e 

---

# Supplementary document: `notebooks/eda/README.md`

# Notebook 01 — Exploratory Data Analysis

**File:** `1-ids-eda.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/1-ids-eda)
**Plots:** [`/plots/eda/`](../../plots/eda/)

---

## What this notebook does

The raw NF-UQ-NIDS-v2 CSV is ~76 million rows and does not fit in memory at once. This notebook reads it in chunks of 500,000 rows, applies stratified sampling (33% per class per chunk), and builds a representative in-memory subset for analysis.

### Steps

1. **Environment check** — GPU, CPU, RAM, Python and library versions
2. **Chunked loading** — 500k-row chunks, deduplication per chunk, 33% stratified sample
3. **Data quality**
   - Shape, dtypes, missing value counts
   - Infinity value detection and replacement
   - Corrupted label check (malformed binary/multiclass targets)
   - Duplicate row count
   - Constant feature detection
4. **Class distribution**
   - Binary (Benign vs Attack) — bar + pie chart
   - Multiclass (21 classes) — horizontal bar chart with counts and percentages
   - Rare class report (classes with fewer than 1,000 samples)
5. **Statistical profiling**
   - `.describe()` — mean, std, min/max, percentiles
   - Skewness and kurtosis per feature
   - Features flagged for log transform (|skew| > 1.0)
6. **Visualisations**
   - Feature histograms (original scale)
   - Feature histograms (log1p scale for high-skew features)
   - Skewness and kurtosis bar charts
   - Correlation matrix (50k-row sample, full 42-feature heatmap)
7. **Multicollinearity analysis**
   - Pairs with Pearson correlation > 0.95
   - Weakest-of-pair dropped based on target correlation
8. **Mutual information** — MI scores vs binary target, top features ranked
9. **Feature distributions by attack type** — violin/strip plots for top 6 MI features
10. **PCA** — 2-component projection, Benign vs Attack, variance explained

### Outputs saved

All plots are saved to `/kaggle/working/plots_eda/` and zipped for download.

---

## Key findings

- **76.4M total records**, reduced to ~25M after 33% stratified sampling
- **3 dominant classes** (Benign, DDoS, DoS) account for ~85% of traffic
- **8 rare classes** with fewer than 5,000 samples — significant class imbalance
- **All features are numerical** — no categorical columns, no text
- **~35% of features have |skew| > 1.0** and benefit from log1p transform
- **Multiple highly correlated pairs** identified (e.g. MAX_IP_PKT_LEN / LONGEST_FLOW_PKT)
- **PCA explains only 27.3% variance in 2 components** — high intrinsic dimensionality, non-linear separability
- **Top MI features:** `LONGEST_FLOW_PKT`, `MAX_IP_PKT_LEN`, `IN_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `MIN_TTL`, `MAX_TTL`
-e 

---

# Supplementary document: `notebooks/preprocessing/README.md`

# Notebook 02 — Preprocessing & Feature Engineering

**File:** `2-ids-preprocessing.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/2-ids-preprocessing)
**Plots:** [`/plots/preprocessing/`](../../plots/preprocessing/)

---

## What this notebook does

Transforms the raw sampled data into clean, scaled, split arrays ready for model training. All artefacts (scaler, encoder, arrays) are saved and reused by every downstream model notebook.

### Steps

1. **Chunked loading** — same pipeline as EDA (33% stratified sample, deduplication)
2. **Data cleaning**
   - Replace ±Inf with NaN
   - Median imputation (fitted on full sample, saved as `imputer.pkl`)
3. **Feature engineering** — 12 new domain-specific features added on top of the 39 original features

| Feature | Description |
|---------|-------------|
| `bytes_per_pkt_in` | Avg payload size per inbound packet — small = flood/scan |
| `bytes_per_pkt_out` | Avg payload size per outbound packet |
| `upload_ratio` | Outbound bytes / total bytes — high in exfiltration |
| `byte_asymmetry` | Absolute difference between in/out bytes |
| `pkt_asymmetry` | Absolute difference between in/out packets |
| `is_http_https` | Flag: destination port 80 or 443 |
| `is_ssh_telnet` | Flag: destination port 22 or 23 |
| `is_dns` | Flag: destination port 53 |
| `ttl_range` | MAX_TTL − MIN_TTL — path diversity proxy |
| `syn_only_flag` | TCP_FLAGS == 0x02 — SYN without ACK (scan indicator) |
| `retransmit_pkt_ratio` | Retransmitted packets / total packets |
| `flow_duration_ms` | Flow duration in milliseconds |

4. **Multicollinearity pruning** — 9 features dropped (highest-correlated pairs where the weaker member has lower MI with target):
   `MAX_IP_PKT_LEN`, `ICMP_IPV4_TYPE`, `MAX_TTL`, `CLIENT_TCP_FLAGS`, `LONGEST_FLOW_PKT`, `RETRANSMITTED_OUT_PKTS`, `retransmit_pkt_ratio`, `NUM_PKTS_1024_TO_4096_BYTES`, `SHORTEST_FLOW_PKT`

5. **Label encoding** — `LabelEncoder` fitted on 21 attack class names → saved as `label_encoder.pkl`

6. **Log1p transform** — applied in-place to all features with |skew| > 1.0

7. **Train / Val / Test split** — 60 / 20 / 20, stratified on multiclass target to guarantee rare class representation in every split

8. **RobustScaler** — fitted on training data only, applied to val and test (prevents data leakage)

9. **Class weights** — `compute_class_weight('balanced')` on training labels, saved as `class_weights.pkl`; log-smoothed version also saved for use with neural network models

10. **Mutual information** — computed on 200k training samples, column order saved as `mi_selected_cols.pkl` (feature name list in MI rank order)

11. **Attack-only split** — separate arrays for Model 3 (rows where binary label = 1), saved as `X_train_att.npy` etc.

### Saved artefacts

| File | Contents |
|------|----------|
| `imputer.pkl` | Fitted `SimpleImputer` (median) |
| `scaler.pkl` | Fitted `RobustScaler` |
| `label_encoder.pkl` | Fitted `LabelEncoder` (21 classes) |
| `class_weights.pkl` | Balanced class weight dict (multiclass) |
| `class_weight_att_dict.pkl` | Balanced class weight dict (attack-only) |
| `mi_selected_cols.pkl` | Feature name list in MI rank order |
| `X_train.npy` | Training features (float32) |
| `X_val.npy` | Validation features |
| `X_test.npy` | Test features |
| `y_train_bin.npy` | Binary training labels |
| `y_val_bin.npy` | Binary validation labels |
| `y_test_bin.npy` | Binary test labels |
| `y_train_multi.npy` | Multiclass training labels (encoded) |
| `y_val_multi.npy` | Multiclass validation labels |
| `y_test_multi.npy` | Multiclass test labels |
| `X_train_att.npy` | Attack-only training features |
| `X_val_att.npy` | Attack-only validation features |
| `X_test_att.npy` | Attack-only test features |
| `y_train_att_multi.npy` | Attack-only multiclass training labels |
| `y_val_att_multi.npy` | Attack-only multiclass validation labels |
| `y_test_att_multi.npy` | Attack-only multiclass test labels |

---

## Split sizes

| Split | Rows | Features |
|-------|------|---------|
| Train | ~13.7M → subsampled to ~5.2M in DL models notebooks | 42 |
| Val | ~4.6M | 42 |
| Test | ~4.6M | 42 |
-e 

---

# Supplementary document: `notebooks/xgboost/README.md`

# Notebook 03 — XGBoost

**File:** `xgboost.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/xgboost)
**Plots:** [`/plots/xgboost/`](../../plots/xgboost/)

---

## Overview

XGBoost with GPU acceleration (`tree_method='hist', device='cuda'`). Serves as the primary tree-based baseline. Fast to train, strong out-of-the-box, and highly interpretable via feature gain importance.

Class imbalance is handled with log-smoothed sample weights passed to the `sample_weight` parameter.

---

## Hyperparameters

### Model 1 — Binary

```python
{
    'max_depth'        : 6,
    'learning_rate'    : 0.05,
    'n_estimators'     : 1000,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'min_child_weight' : 5,
    'tree_method'      : 'hist',
    'device'           : 'cuda',
    'eval_metric'      : ['logloss', 'auc'],
    'early_stopping_rounds': 50,
}
```

### Model 2 — Full Multiclass (21 classes)

```python
{
    'max_depth'        : 8,
    'learning_rate'    : 0.03,
    'n_estimators'     : 1500,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.6,
    'min_child_weight' : 10,
    'tree_method'      : 'hist',
    'device'           : 'cuda',
    'eval_metric'      : ['mlogloss'],
    'early_stopping_rounds': 50,
}
```

### Model 3 — Attack-Only Multiclass (20 classes)

```python
{
    'max_depth'        : 8,
    'learning_rate'    : 0.03,
    'n_estimators'     : 1500,
    'subsample'        : 0.75,
    'colsample_bytree' : 0.7,
    'min_child_weight' : 10,
    'tree_method'      : 'hist',
    'device'           : 'cuda',
    'eval_metric'      : ['mlogloss'],
    'early_stopping_rounds': 50,
}
```

---

## Evaluation plots

Each model produces:
- Learning curve (train loss + val loss)
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Feature importance bar chart (top 20, engineered features highlighted in red)
- Comparison bar chart vs published papers on NF-UQ-NIDS-v2
-e 

---

# Supplementary document: `notebooks/catboost/README.md`

# Notebook 04 — CatBoost

**File:** `catboost.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/catboost)
**Plots:** [`/plots/catboost/`](../../plots/catboost/)

---

## Overview

CatBoost with GPU acceleration. Uses symmetric trees (oblivious decision trees) and ordered boosting, which reduce overfitting on imbalanced data compared to standard gradient boosting. Training uses a `Pool` object with explicit sample weights.

Early stopping is handled natively via `od_type='Iter'` and `od_wait` on a validation pool — no manual loop required.

---

## Hyperparameters

### Model 1 — Binary

```python
{
    'iterations'          : 1500,
    'depth'               : 6,
    'learning_rate'       : 0.05,
    'l2_leaf_reg'         : 5.0,
    'random_strength'     : 0.1,
    'bagging_temperature' : 0.5,
    'border_count'        : 128,
    'task_type'           : 'GPU',
    'loss_function'       : 'Logloss',
    'eval_metric'         : 'AUC',
    'od_type'             : 'Iter',
    'od_wait'             : 50,
}
```

### Model 2 — Full Multiclass (21 classes)

```python
{
    'iterations'          : 1500,
    'depth'               : 9,
    'learning_rate'       : 0.03,
    'l2_leaf_reg'         : 15.0,
    'random_strength'     : 2.0,
    'bagging_temperature' : 1.0,
    'border_count'        : 128,
    'task_type'           : 'GPU',
    'loss_function'       : 'MultiClass',
    'eval_metric'         : 'MultiClass',
    'od_type'             : 'Iter',
    'od_wait'             : 50,
}
```

### Model 3 — Attack-Only Multiclass (20 classes)

```python
{
    'iterations'          : 1200,
    'depth'               : 7,
    'learning_rate'       : 0.04,
    'l2_leaf_reg'         : 10.0,
    'random_strength'     : 1.5,
    'bagging_temperature' : 0.8,
    'border_count'        : 128,
    'task_type'           : 'GPU',
    'loss_function'       : 'MultiClass',
    'eval_metric'         : 'MultiClass',
    'od_type'             : 'Iter',
    'od_wait'             : 50,
}
```

---

## Notes on CatBoost vs XGBoost for this dataset

- CatBoost's ordered boosting prevents target leakage during training, which matters for rare classes
- `l2_leaf_reg` replaces XGBoost's `min_child_weight` for regularisation
- `bagging_temperature` controls Bayesian bootstrap intensity (higher = more random)
- Feature importance from CatBoost uses `get_feature_importance()` which returns PredictionValuesChange scores

---

## Evaluation plots

Each model produces:
- Learning curve with best iteration marker
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Feature importance bar chart (top 20, engineered features highlighted)
- Comparison bar chart vs published papers on NF-UQ-NIDS-v2
-e 

---

# Supplementary document: `notebooks/tabnet/README.md`

# Notebook 05 — TabNet

**File:** `tabular-network.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/tabular-network)
**Plots:** [`/plots/tabnet/`](../../plots/tabnet/)

---

## Overview

TabNet (Arik & Pfister, 2021) is an attention-based deep learning architecture designed specifically for tabular data. Unlike standard MLPs, TabNet selects a sparse subset of features at each decision step using a sequential attention mechanism, making it inherently interpretable.

The key advantage over tree models shown in this notebook: **the attention masks reveal which features each attack type focuses on**, producing a per-class feature importance heatmap that trees cannot replicate.

Because the full 13.7M training rows exceed Kaggle's memory budget, the training set is subsampled to ~5.2M rows by capping the three largest classes (Benign, DDoS, DoS at 1.2M each; Recon, Scanning, Mirai at 400k each) while keeping all rare class samples intact.

---

## Hyperparameters (shared across all three models)

```python
TABNET_PARAMS = {
    'n_d'               : 16,    # width of decision step output
    'n_a'               : 16,    # width of attention embedding
    'n_steps'           : 3,     # number of sequential decision steps
    'gamma'             : 1.3,   # attention sparsity regularisation
    'n_shared'          : 2,     # shared layers across steps
    'lambda_sparse'     : 1e-4,  # sparsity loss weight
    'optimizer_fn'      : Adam,
    'optimizer_params'  : {'lr': 2e-3},
    'scheduler_fn'      : StepLR,
    'scheduler_params'  : {'step_size': 10, 'gamma': 0.9},
    'mask_type'         : 'sparsemax',
    'device_name'       : 'cuda',
    'seed'              : 42,
    'verbose'           : 1,
}
```

**Fit parameters:**

| Param | Binary | Multiclass | Attack-Only |
|-------|--------|-----------|-------------|
| max_epochs | 100 | 100 | 100 |
| patience | 15 | 15 | 15 |
| batch_size | 16384 | 16384 | 16384 |
| virtual_batch_size | 512 | 512 | 512 |

---

## Memory management

TabNet's `predict()` is called in chunks of 131,072 rows to avoid OOM on the 4.5M-row test set. Training data is deleted and CUDA cache cleared between models.

---

## Unique visualisation: attention mask analysis

After all three models are trained, `tabnet_multi` is reloaded and `explain()` is called on a 2,000-sample subset of the attack-only test set. This produces:

1. **Mean attention bar chart** — average attention weight per feature across all samples
2. **Per-class attention heatmap** — mean attention per feature grouped by attack class, showing that TabNet learns distinct feature subsets for different attack types

---

## Evaluation plots

Each model produces:
- Learning curve (train loss + val AUC/accuracy)
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Feature importance (TabNet `feature_importances_`, top 20)
- Attention mask visualisation (mean + per-class heatmap)
- Comparison bar chart vs published papers
-e 

---

# Supplementary document: `notebooks/residual-mlp/README.md`

# Notebook 06 — Residual MLP

**File:** `residual-mlp.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/residual-mlp)
**Plots:** [`/plots/residual-mlp/`](../../plots/residual-mlp/)

---

## Overview

A deep MLP with residual connections, BatchNorm, and GELU activations — trained from scratch using PyTorch.

The architecture follows a simple principle: stack residual blocks wide rather than deep, let BatchNorm handle internal covariate shift, and use AMP fp16 to double throughput on the T4 GPU.

**All three models finish training in under 40 minutes on Kaggle free T4.**

---

## Architecture

```
Input (42 features)
    │
    └── BatchNorm1d(42)
         │
         └── ResBlock × 4
              ├── Linear(in → 512) → BN → GELU → Dropout(0.15)
              ├── Linear(512 → 512) → BN → GELU → Dropout(0.15)
              └── skip: Linear(in → 512) [only on first block]
         │
         └── Linear(512 → n_classes)
```

Total parameters: ~1.5M (binary) / ~1.6M (21-class)

---

## Hyperparameters

```python
RESMLP_PARAMS = {
    'hidden_dim'   : 512,
    'n_blocks'     : 4,
    'dropout'      : 0.15,
    'lr'           : 3e-4,
    'weight_decay' : 1e-4,
}

FIT_PARAMS = {
    'max_epochs'  : 20,
    'patience'    : 5,
    'batch_size'  : 8192,
    'num_workers' : 2,
}
```

**Training details:**
- Optimiser: AdamW
- LR schedule: Linear warmup (3 epochs) → cosine annealing to 0
- Gradient clipping: max norm 1.0
- Mixed precision: `torch.cuda.amp.GradScaler` (fp16)
- Early stopping: patience on primary validation metric

---

## Feature importance

TabNet has built-in attention masks. This model uses **gradient-based feature importance** instead: backpropagates through the model on a stored training sample and takes the mean absolute gradient per input dimension. The resulting importance scores are plotted identically to the other notebooks for direct comparison.

The `explain(X)` method returns per-sample gradient magnitudes with the same shape as TabNet's `explain()` output, enabling the same per-class heatmap visualisation.

---

## Why Residual MLP over a plain MLP?

- Skip connections allow gradients to flow cleanly through depth without vanishing
- BatchNorm before each activation stabilises training on the heavily skewed, multi-scale NetFlow features
- With 5M training rows, the model learns quickly — depth adds more than width here
- The residual structure makes the model robust to the extreme class imbalance via the log-smoothed sample weights, because rare class gradients are not washed out

---

## Evaluation plots

Each model produces:
- Learning curve (train loss + val AUC/accuracy/balanced accuracy)
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Gradient feature importance bar chart (top 20, engineered features highlighted)
- Per-class gradient importance heatmap (same interface as TabNet attention)
- Comparison bar chart vs published papers
