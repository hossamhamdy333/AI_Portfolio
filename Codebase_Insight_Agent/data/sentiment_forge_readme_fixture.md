<div align="center">

# Sentiment Forge — Three Models, One Test Set

`scikit-learn` `torch` `transformers` `onnx` `huggingface_hub` `MLflow` `DVC` `pandas` `PyYAML` `pytest`

</div>

---

### Contents

- [Summary](#summary)
- [Problem & motivation](#problem--motivation)
- [Approach](#approach)
- [Data](#data)
- [Results](#results)
- [What I'd do differently / limitations](#what-id-do-differently--limitations)
- [Stack](#stack)

---

## Summary

**Headline result** (same 2,204-sentence test set for every model):

- Best model: fine-tuned BERT-base, with weighted F1 0.5136, macro F1 0.5142 and AUC-ROC 0.8361.
- TF-IDF + LogReg: weighted F1 0.4171, macro F1 0.4107, AUC-ROC 0.7410.
- BiLSTM + GloVe: weighted F1 0.4140, macro F1 0.4122, AUC-ROC 0.7564.

A 5-class sentiment classifier for single-sentence movie reviews (SST-5),
built three separate ways — TF-IDF + Logistic Regression, a BiLSTM over
GloVe embeddings, and a fully fine-tuned `bert-base-uncased` — and scored
head-to-head on the identical 2,204-sentence held-out test set by the
identical `compute_metrics()` function. BERT wins by roughly 10 F1 points
(0.5136 weighted F1 vs. 0.4171 and 0.4140) and by a wider margin on AUC-ROC
(0.8361 vs. 0.7410 and 0.7564). The more interesting number is the one
between the two non-transformer models: the BiLSTM, with pretrained
embeddings, 4M parameters and a GPU training loop, lands 0.003 F1 *below*
a bag-of-words linear model, and 0.0015 above it on macro F1 — a tie, in
other words, for a large difference in engineering cost. The best model is
exported to ONNX (verified against the PyTorch logits, max absolute
difference 7.82e-4) and hosted on the Hugging Face Hub, so it can be run
without retraining anything.

Two numbers in the previous version of this README don't survive a check
against the committed notebook outputs, and both are corrected below: the
latency/size table, and which BERT checkpoint the head-to-head comparison
actually loaded. See Results and Limitations.

## Problem & motivation

SST-5 is a harder task than the binary sentiment benchmarks it's usually
confused with. Five ordered classes (very negative → very positive) on
single sentences averaging 19 words means the label boundaries are narrow
and partly subjective: the difference between "negative" and "very
negative" is a matter of degree that human annotators themselves disagree
on. Absolute numbers here look low next to binary SST-2 accuracy in the
90s, and that's a property of the task, not a sign of a broken pipeline.

The naive framing of a project like this is "fine-tune BERT, report the
number, done." Two things make that undersell what's actually worth
knowing.

First, "which model should I use" is a deployment question, not just an
accuracy question, and it can't be answered from published numbers across
different papers and setups. The only way to answer it is to train the
alternatives on byte-identical splits and score them with the same
function on the same rows. That's what makes the TF-IDF-vs-BiLSTM result
above meaningful: it isn't a claim that LSTMs are bad, it's a measurement
that on 19-word sentences there is little sequential structure left for a
recurrent model to exploit that a word + character n-gram representation
hasn't already captured — so the extra capacity buys ~0 F1, 5x the model
size, and a GPU dependency.

Second, the vocabulary is long-tailed in a way that shapes every modeling
choice: 16,412 unique tokens in the training set, of which 8,235 (50.2%)
appear exactly once. Half the vocabulary is effectively unlearnable from
this data alone. That's the direct argument for pretrained representations
(GloVe, then BERT's subword vocabulary) over anything learned from
scratch, and the argument for adding character n-grams to the TF-IDF
baseline rather than relying on word n-grams that can't generalize to an
unseen word at all.

## Approach

Every hyperparameter lives in `configs/config.yaml`, loaded by all five
notebooks, so nothing is defined twice across the pipeline. The shared
data-cleaning and metrics code lives in `src/`; each model gets its own
numbered notebook.

### Cleaning

(`src/data_utils.py`, applied in `notebooks/01_eda.ipynb`):
`clean_dataframe()` drops exact-duplicate sentences, lowercases, collapses
runs of whitespace, and drops sentences under `min_length = 3` words.
Punctuation is deliberately kept — it carries sentiment signal, and the
character n-gram vectorizer can use it. Order matters here and is worth
stating: deduplication happens on the raw text, before lowercasing, so
two sentences differing only in case would both survive.

### 1. TF-IDF + Logistic Regression

(`notebooks/02_classical.ipynb`, `src/features.py`). A `FeatureUnion` of
two vectorizers rather than one:
word-level 1–2 grams (`max_features=20000`, `min_df=2`, `max_df=0.95`,
`sublinear_tf=True`, with a `token_pattern` that requires at least two
letters) plus character-level `char_wb` 3–5 grams (`max_features=20000`,
`min_df=3`). Word n-grams catch sentiment phrases ("not good", "the
worst"); character n-grams catch morphology and give the model something
to work with on the 50% of the vocabulary that appears once. The
classifier is multinomial `LogisticRegression` with
`class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000`. `C` is the
only tuned parameter, via 5-fold `GridSearchCV` on `f1_weighted` over
`[0.01, 0.1, 1.0, 10.0, 100.0]`.

### 2. BiLSTM + GloVe

(`notebooks/03_BiLSTM.ipynb`, `src/models.py`). A vocabulary is built from
the training split only (16,572 entries including
`<PAD>` and `<UNK>`), and initialized from `glove.6B.100d`, which covers
15,279 of those tokens (92.2%); the rest are drawn from
`Uniform(-0.25, 0.25)` with `<PAD>` forced to zeros. The model is a
2-layer bidirectional LSTM, hidden dim 256, dropout 0.3 on both the
embedding output and the pooled representation, classifying from the
concatenated final forward and backward hidden states (512-dim) — 3,969,909
parameters total. Sequences are truncated and padded to a fixed 64 tokens;
since the longest training sentence is 52 words, truncation never actually
binds on train. Trained 10 epochs, Adam at 1e-3, batch 64, gradient-norm
clipping at 1.0, and a class-weighted cross-entropy using
`sklearn.utils.class_weight.compute_class_weight("balanced")`, which gives
`[1.567, 0.770, 1.056, 0.734, 1.325]`. The checkpoint kept is the best
epoch by **validation F1**, not by validation loss or last epoch — a
distinction that matters here, because validation loss bottoms out at
epoch 3 and rises steadily afterward while validation F1 keeps improving
through epoch 7.

### 3. BERT fine-tuning

(`notebooks/04_BERT-Fine-tuning.ipynb`). `bert-base-uncased`, all
109,486,085 parameters trainable — fully
fine-tuned, not frozen or probed. 5 epochs, batch size 32, LR 2e-5,
`warmup_ratio=0.1`, `weight_decay=0.01`, fp16, seed 42, evaluation and
checkpointing every epoch with `load_best_model_at_end=True` selecting on
`f1_weighted`. Class weighting is applied through a `WeightedTrainer`
subclass overriding `compute_loss`, because the stock `Trainer` has no
hook for a weighted loss. Note that its weights are hardcoded as
`[2.0, 1.3, 1.8, 1.3, 2.0]` — hand-chosen, not the computed balanced
weights the BiLSTM uses, so the two models are not weighted identically
(see Limitations).

**Export and serving**: the fine-tuned model is exported with
`torch.onnx.export` at opset 18, with dynamic axes on batch and sequence
length so it isn't locked to one input shape, and the ONNX logits are
compared numerically against the PyTorch model's on the same input before
the artifact is trusted. The ONNX model plus tokenizer are served from the
Hugging Face Hub (`hossam3759180/sentiment_forge`, set in
`config.yaml`'s `serving.model_repo`) and pulled with `snapshot_download`
at inference time.

**Versioning and tracking**: model binaries and the three parquet splits
are DVC-tracked (only `.dvc` pointer files are committed); MLflow logs
metrics to a DagsHub-hosted tracking server; Weights & Biases tracks the
per-epoch training curves for the BiLSTM and BERT runs. 7 pytest unit
tests cover the cleaning functions and the metrics function, run by GitHub
Actions on every push that touches `sentiment_forge/**`.

## Data

Source: SST-5 (Stanford Sentiment Treebank, 5-class) via the `SetFit/sst5`
Hugging Face dataset. Splits come from the dataset itself, not from a
random re-split, so they match what other SST-5 results are measured on.

| Split | Raw rows | After cleaning | Duplicates found | Mean words |
|---|---:|---:|---:|---:|
| Train | 8,544 | 8,494 | 10 | 19.14 |
| Validation | 1,101 | 1,098 | 1 | 19.32 |
| Test | 2,210 | 2,204 | 0 | 19.19 |

No nulls in any split. The 50 rows dropped from train are the 10 duplicates
plus 40 sentences under 3 words; 136 training sentences (1.59%) have 3 words
or fewer, and those at exactly 3 are kept, because most are still real
signal (*"too bad ."*, *"stay away ."*, *"delirious fun ."*). Word counts
run 2–52 in train, median 18.

Class balance is moderately skewed but not severe — train counts (raw):
positive 2,322, negative 2,218, neutral 1,624, very positive 1,288, very
negative 1,092, a 2.13:1 ratio between largest and smallest. Test-set
support after cleaning: negative 632, positive 510, very positive 399,
neutral 385, very negative 278. This imbalance is why class weighting
appears in all three models and why weighted F1 is the primary metric in
`config.yaml`, with macro F1 reported alongside it so a model can't win by
serving the majority classes.

Vocabulary: 146,121 total tokens, 16,412 unique, 8,235 singletons. Mean
sentence length is essentially flat across classes (18.48–19.73 words), so
length itself carries no class signal.

## Results

All three models scored on the same 2,204-row test set by
`src/evaluate.py`'s `compute_metrics()`, from
`notebooks/05_comparison.ipynb`:

| Model | F1 (weighted) | F1 (macro) | AUC-ROC (OvR) |
|---|---:|---:|---:|
| TF-IDF + LogReg | 0.4171 | 0.4107 | 0.7410 |
| BiLSTM + GloVe | 0.4140 | 0.4122 | 0.7564 |
| **BERT-base (fine-tuned)** | **0.5136** | **0.5142** | **0.8361** |

The BERT row needs a caveat the notebook doesn't state. `04_BERT-Fine-tuning.ipynb`
evaluates the best-by-validation-F1 model that `load_best_model_at_end`
restored, and reports **0.5076 / 0.5081 / 0.8346**. `05_comparison.ipynb`
re-loads BERT from disk with `sorted(glob(f"{ckpt_dir}/checkpoint-*"))[-1]`
— a lexicographic sort, not a numeric one. With 266 optimizer steps per
epoch, the five checkpoints are `checkpoint-266` … `checkpoint-1330`, and
string-sorting them puts `checkpoint-798` (epoch 3) last. So the two
notebooks report two different sets of weights, and the per-class numbers
confirm it (very-negative precision 0.43 in notebook 04 vs. 0.51 in
notebook 05). Both land at ≈0.51 weighted F1, so the headline comparison
holds either way, but the comparison table above is not scoring the
checkpoint that was exported to ONNX and published — notebook 04's is.

### Per-class F1

(test set; BERT column from `05_comparison.ipynb`):

| Class | TF-IDF | BiLSTM | BERT |
|---|---:|---:|---:|
| very negative | 0.37 | 0.36 | 0.53 |
| negative | 0.44 | 0.41 | 0.50 |
| neutral | 0.28 | 0.33 | 0.37 |
| positive | 0.41 | 0.39 | 0.54 |
| very positive | 0.54 | 0.57 | 0.64 |

"Neutral" is the weakest class for every model and "very positive" the
strongest for every model. The BiLSTM's neutral behavior is worth noting
specifically: 0.44 recall at 0.26 precision, the only model that
over-predicts neutral rather than under-predicting it. Errors overall
concentrate between adjacent classes, which is the failure mode you'd
expect on an ordered scale.

Total error counts on the test set: TF-IDF 1,290/2,204 (58.5%), BiLSTM
1,303/2,204 (59.1%), BERT 1,082/2,204 (49.1%, from notebook 04's
best-checkpoint run).

### Tuning and training detail worth keeping

- Grid search over `C` (5-fold CV, weighted F1 on train):
  0.01 → 0.3534, 0.1 → 0.3782, **1.0 → 0.4004**, 10.0 → 0.3883,
  100.0 → 0.3809. The curve is unimodal with a clear peak, so the grid is
  wide enough to have actually bracketed the optimum rather than stopping
  at an edge.
- BiLSTM validation F1 by epoch: 0.3182, 0.3871, 0.3715, 0.3594, 0.4139,
  0.4199, **0.4345**, 0.4058, 0.4151, 0.4247. Validation loss over the same
  epochs: 1.4223, 1.3676, **1.3100**, 1.3194, 1.3363, 1.3822, 1.4867,
  1.6255, 1.6418, 1.8860. Train loss falls monotonically 1.5094 → 0.6477.
  The model is clearly overfitting from epoch 4 on by loss, while still
  improving on the metric that was selected on.
- **Open item:** BERT per-epoch validation F1 — the training cell's output is saved
  as an HTML widget object, so the numbers aren't recoverable from the
  committed notebook; they're in the `bert_base_uncased` W&B run.

### Same five sentences through all three models

(from
`05_comparison.ipynb`), which is where the representational difference
becomes concrete:

| Sentence | TF-IDF | BiLSTM | BERT |
|---|---|---|---|
| "a baffling misfire , and possibly the weakest movie woody allen has made" | very negative | negative | very negative |
| "the editing is chaotic , the photography grainy and badly focused" | very negative | very negative | very negative |
| "my oh my , is this an invigorating , electric movie" | **very negative** | very positive | very positive |
| "eight crazy nights is a showcase for sandler s many talents" | **very negative** | neutral | positive |
| "the movie is a blast of educational energy" | very positive | very positive | very positive |

Rows 3 and 4 are the interesting ones. Both are positive sentences that
TF-IDF calls *very negative*, and both fail for the same reason: the
sentiment lives in word order and in words that were rare in training
("invigorating", "electric"), neither of which a bag of n-grams can use.
The top-weighted word features per class show the same thing from the other
direction — TF-IDF's strongest "neutral" features are `but`, `yet`,
`but not`, `not`, `though`, i.e. the model has learned that hedging
conjunctions mean mixed sentiment, which is real signal, but it's the only
kind of structure available to it.

### Latency and model size

The numbers previously published
in this README
(0.53 / 0.40 / 2.81 ms and ~3 / ~15 / 440 MB) do not match what the
committed notebook actually printed, so here is what the repo supports:

| Model | Latency (measured) | On-disk size |
|---|---:|---:|
| TF-IDF + LogReg | 0.31 ms/sample | 3,065,092 B (2.9 MB) |
| BiLSTM + GloVe | 0.37 ms/sample | 15,883,347 B (15.1 MB) |
| BERT-base (ONNX) | 2.52 ms/sample | 439,301,420 B across 4 files (419 MB) |

The size cell in `05_comparison.ipynb` prints **0.7 MB** for BERT, because
it measures only `model.onnx` from the downloaded Hub snapshot. The real
figure is in `outputs/models/bert_base_onnx.dvc`, which records
439,301,420 bytes over 4 files for the export directory — consistent with
109.5M fp32 parameters, and consistent with the ONNX weights living in a
separate external-data file that the size cell never looks at. So BERT is
roughly 140x the classical model on disk and ~8x its per-sample latency,
not the 0.7 MB the notebook's own table claims.

Two caveats on the latency column: it's a single un-repeated wall-clock
measurement including the first-batch warm-up, and it isn't measured on
equal hardware — the BiLSTM and BERT run on a Colab GPU, TF-IDF on CPU. It
supports "BERT is roughly an order of magnitude slower" and nothing
finer-grained than that.

**ONNX fidelity**: max absolute difference between ONNX Runtime and
PyTorch logits on the same input, 0.0007817745.

### Hosted model check

(`05_comparison.ipynb`, pulling from the Hub, not from
local weights):

```
this movie was absolutely wonderful and touching → very positive (0.955)
terrible film , complete waste of time           → very negative (0.918)
it was okay , nothing special                    → neutral (0.632)
```

**Tests**: 7 unit tests in `tests/test_pipeline.py` (3 on `clean_text`,
2 on `clean_dataframe`'s dedupe and length filter, 2 on `compute_metrics`).
I installed the dependencies and ran them directly — all 7 pass. They cover
the deterministic parts of the pipeline only; no test asserts anything
about model quality.

## What I'd do differently / limitations

- **The checkpoint-selection bug in `05_comparison.ipynb` is the most
  concrete thing to fix here.** `sorted(glob("checkpoint-*"))[-1]` sorts as
  strings, so it picks `checkpoint-798`, not the highest-step or the
  best-by-validation checkpoint. The fix is either
  `max(..., key=lambda p: int(p.split("-")[-1]))` or, better, saving the
  best model to a fixed path at the end of notebook 04 so the comparison
  notebook never has to guess. As it stands, the head-to-head table scores
  a different BERT than the one that was exported and published.
- **`src/evaluate.py` is missing `error_analysis()`, which notebooks 02,
  03, and 04 all import from it.** Those notebooks ran because they write
  their own copy of `evaluate.py` into `src/` at runtime, but the version
  actually committed to the repo defines only `compute_metrics` and
  `plot_confusion_matrix`. A fresh clone hits an `ImportError` on the error
  analysis cell. Same class of problem in a second place: `build_vocab()`
  exists only inside notebooks 03 and 05, defined twice, rather than once
  in `src/`.
- **The BiLSTM's vocabulary is never serialized with its weights.**
  `bilstm.pt` is a bare `state_dict`; `05_comparison.ipynb` reconstructs the
  vocab by re-running `build_vocab()` over `train.parquet`. That works only
  as long as the training data file and the tokenization function both stay
  byte-identical — if either changes, every embedding index silently shifts
  and the model produces confident nonsense with no error raised. The vocab
  should be saved next to the checkpoint.
- **BERT and the BiLSTM are not weighted the same way**, despite being
  presented as a controlled comparison. The BiLSTM uses sklearn's computed
  balanced weights (`[1.567, 0.770, 1.056, 0.734, 1.325]`); `WeightedTrainer`
  uses hand-picked `[2.0, 1.3, 1.8, 1.3, 2.0]`. Nothing in the repo tests
  what BERT scores under the computed weights, so part of its margin could
  be a better-tuned loss rather than a better representation. The data
  splits and the metric function *are* identical, so the comparison is still
  meaningful — just not as clean as "same everything but the architecture."
- **MLflow does not cover all three models, despite being described as the
  cross-run comparison layer.** Only `02_classical.ipynb` and
  `05_comparison.ipynb` log to it; the BiLSTM and BERT training runs report
  to W&B only. So the per-model training runs aren't queryable in one place
  — only the final comparison run is.
- **The DVC remote is a local Google Drive path**
  (`/content/drive/MyDrive/AI_Portfolio_DVC`), added inline in each
  notebook. Nobody else can `dvc pull` this project, and
  `02_classical.ipynb`'s own output shows a `Some of the cache files do not
  exist neither locally nor on remote` warning with 8 missing hashes, so
  even the author's remote isn't complete. A shared remote (DagsHub already
  hosts the MLflow server) would make the data and model artifacts actually
  reproducible.
- **No committed code uploads the ONNX model to the Hub.** The artifact is
  live and `snapshot_download` pulls it successfully, but the push step
  happened outside the five notebooks, so there's no reproducible path from
  a training run to the published model.
- **Neutral is the weak class in all three models and nothing here
  targets it.** Focal loss, an ordinal-regression objective (which would
  use the fact that the labels are ordered — none of these three models
  does), or a two-stage coarse-then-fine classifier are the obvious things
  to try, and none is tested.
- **No smaller transformer was benchmarked.** DistilBERT or MiniLM would
  answer the actual deployment question this project raises — how much of
  BERT's 10-point gain survives at a fraction of 419 MB and 2.52 ms — and
  would make the accuracy/cost curve three points instead of a binary
  choice. Int8 quantization of the existing ONNX export is the cheaper
  version of the same experiment.
- **Single run, single seed, everywhere.** Seeds are fixed (42) so each run
  is reproducible, but nothing is repeated across seeds, so there's no
  variance estimate on any number here. The 0.0031 F1 gap between TF-IDF
  and the BiLSTM is almost certainly inside run-to-run noise, which is why
  it's described above as a tie rather than a win for either.
- **`LogisticRegression(multi_class="multinomial")` is deprecated** in
  scikit-learn ≥1.5 and removed in 1.7; `requirements.txt` pins 1.4.2, so
  it works today and breaks on a dependency bump. The parameter can simply
  be dropped — multinomial is the default for `lbfgs`.
- **`01_eda.ipynb` writes the parquet splits twice**, first the raw
  DataFrames and then, further down, the cleaned ones. Correct only if the
  whole notebook runs top to bottom; a partial run leaves uncleaned data as
  the DVC-tracked artifact every downstream notebook loads.

## Stack

- `scikit-learn` 1.4.2 — `TfidfVectorizer`, `FeatureUnion`, `Pipeline`,
  `LogisticRegression`, `GridSearchCV`, `compute_class_weight`, and all
  metrics (`f1_score`, `roc_auc_score`, `confusion_matrix`,
  `classification_report`, `precision_recall_curve`)
- `torch` 2.2.0 (`nn.LSTM`, `nn.Embedding`, `clip_grad_norm_`) for the
  BiLSTM; GloVe `glove.6B.100d` from Stanford NLP for the embedding matrix
- `transformers` 4.40.0 — `AutoTokenizer`,
  `AutoModelForSequenceClassification`, `TrainingArguments`,
  `DataCollatorWithPadding`, and a `Trainer` subclass for weighted loss
- `datasets` 2.19.0 for loading `SetFit/sst5`
- `onnx` 1.16.0 / `onnxruntime` 1.17.3 / `optimum` 1.19.1 for the ONNX
  export and the numerical fidelity check
- `huggingface_hub` 0.23.0 (`snapshot_download`) for pulling the served
  model
- `MLflow` 2.12.1, hosted on DagsHub
  (`https://dagshub.com/hossam3759180/AI_Portfolio.mlflow`)
- `Weights & Biases` for BiLSTM and BERT training curves (project
  `text_classification`)
- `DVC` 3.49.0 + `dvc-gdrive` 3.0.1 for dataset and model-binary versioning
- `pandas` 2.2.2, `numpy` 1.26.4, `matplotlib` 3.8.4, `seaborn` 0.13.2,
  `ydata-profiling` for EDA and figures
- `PyYAML` 6.0.1 — every hyperparameter loaded from `configs/config.yaml`
- `pytest` + GitHub Actions (`.github/workflows/test_sentiment_forge.yml`),
  triggered on pushes touching `sentiment_forge/**`
- Trained on a Colab GPU; served model at
  `hossam3759180/sentiment_forge` on the Hugging Face Hub
