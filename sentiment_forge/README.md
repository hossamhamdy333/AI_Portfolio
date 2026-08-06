# Sentiment Forge

A 5-class sentiment classifier for movie reviews (SST-5), built three separate ways — classical ML, a BiLSTM, and a fine-tuned BERT — and benchmarked head-to-head on the exact same test set. This isn't just a "train a model" project: it goes from raw data all the way to a served API, with the versioning, tracking, and testing you'd actually want around it in a real setting.

```
Raw SST-5 (HuggingFace)
    → EDA (class balance, length, vocab, duplicates)
    → clean_dataframe() [dedupe, drop short sentences, normalize whitespace, lowercase]
    → train / val / test parquet (versioned with DVC)
    → 3 parallel training paths: TF-IDF+LogReg / BiLSTM+GloVe / BERT fine-tune
    → same compute_metrics() on the same held-out test set
    → best model exported to ONNX, verified against the PyTorch output
    → FastAPI /predict endpoint serving the ONNX model
    → pytest + GitHub Actions CI on every push
```

## Why three models instead of one

"Which model should I use" is a real engineering question, not just an academic one — it trades off accuracy against latency and deployment size. So instead of picking one architecture and calling it done, I trained all three on identical data splits and evaluated them the same way, so the comparison numbers below are actually apples-to-apples and not just numbers pulled from different papers.

## The dataset

SST-5 (Stanford Sentiment Treebank, 5-class), pulled from `SetFit/sst5` on HuggingFace.

| Split | Raw size | After cleaning | Avg words/sentence |
|---|---|---|---|
| Train | 8,544 | 8,494 | 19.1 |
| Validation | 1,101 | 1,098 | 19.3 |
| Test | 2,210 | 2,204 | 19.2 |

EDA turned up a few things that shaped the cleaning step: 10 duplicate sentences in train, 1.6% of training sentences at 3 words or fewer (mostly still meaningful, e.g. *"too bad."*, *"stay away."*), and a vocabulary of ~16.4K unique tokens where roughly half the words appear only once — which is exactly the kind of long tail that makes classical bag-of-words models struggle and justifies trying something with pretrained embeddings or contextual representations instead. Classes are moderately imbalanced (negative and positive are the two largest, neutral and very-negative the smallest), which is why class weighting shows up in both the BiLSTM and BERT training runs.

Full EDA — label distribution, sentence-length histograms, per-class vocabulary, an auto-generated `ydata-profiling` report — lives in `notebooks/01_eda.ipynb` and `outputs/figures/`.

## The three models

**1. TF-IDF + Logistic Regression** (`notebooks/02_classical.ipynb`)
Combined word-level (1–2 grams) and character-level (3–5 grams) TF-IDF via `FeatureUnion`, so the model picks up both sentiment phrases ("not good") and morphological/OOV patterns that word n-grams alone would miss. `C` was tuned with 5-fold `GridSearchCV` over `[0.01, 0.1, 1.0, 10.0, 100.0]`; `C=1.0` won with CV F1 of 0.400. `class_weight="balanced"` to offset the class imbalance.

**2. BiLSTM** (`notebooks/03_BiLSTM.ipynb`)
2-layer bidirectional LSTM (hidden dim 256, dropout 0.3) on top of 100-dim GloVe embeddings — 92.2% of the vocab (15,279/16,572 tokens) had a pretrained vector, the rest initialized randomly. ~4M trainable parameters. Trained for up to 10 epochs with a weighted cross-entropy loss (class weights ranging 0.73–1.57) and gradient clipping; best checkpoint was epoch 7 (val F1 0.4345), saved via early best-model checkpointing rather than just taking the last epoch.

**3. BERT fine-tuning** (`notebooks/04_BERT-Fine-tuning.ipynb`)
`bert-base-uncased` (109.5M params) fully fine-tuned — not frozen/probed — for 5 epochs, batch size 32, LR 2e-5 with warmup, weight decay 0.01, mixed precision (fp16), and a custom `WeightedTrainer` subclass to apply the same class weighting during loss computation. Tracked with Weights & Biases.

All three log metrics to MLflow, so the run history and comparison numbers aren't just printed and forgotten — they're queryable later.

## Results

Evaluated on the same 2,204-sentence held-out test set.

| Model | F1 (weighted) | F1 (macro) | AUC-ROC | Latency | Size |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.417 | 0.411 | 0.741 | 0.53 ms/sample | ~3 MB |
| BiLSTM + GloVe | 0.414 | 0.412 | 0.756 | 0.40 ms/sample | ~15 MB |
| **BERT-base (fine-tuned)** | **0.514** | **0.514** | **0.836** | 2.81 ms/sample | 440 MB |

BERT wins on accuracy by a wide margin — about 10 F1 points over the other two, and a meaningfully higher AUC-ROC (0.836 vs ~0.75). What's more interesting is how close TF-IDF and the BiLSTM land: the BiLSTM barely edges out classical ML despite being a "deeper" model, which says a lot about how hard 5-way sentiment actually is on short, single-sentence reviews — there just isn't much sequential structure for an LSTM to exploit that a strong TF-IDF representation doesn't already capture.

Per-class, the "very positive" class is the strongest for every model (F1 in the 0.54–0.65 range), and "neutral" is the weakest across the board (0.28–0.37 F1) — reviews that read as lukewarm or mixed are genuinely ambiguous even to a human labeler, so this isn't a modeling bug so much as a property of the task. Confusion is almost entirely between adjacent classes (negative ↔ very negative, positive ↔ very positive), never between opposite ends of the scale — see the confusion matrices in `outputs/figures/`.

**Same five sentences run through all three models** (from `notebooks/05_comparison.ipynb`), to see where they actually diverge:

| Sentence | TF-IDF | BiLSTM | BERT |
|---|---|---|---|
| "a baffling misfire...the weakest movie woody allen has made" | very negative | negative | very negative |
| "the editing is chaotic, the photography grainy and badly focused" | very negative | very negative | very negative |
| "my oh my, is this an invigorating, electric movie" | very negative ❌ | very positive | very positive |
| "eight crazy nights is a showcase for sandler's many talents" | very negative ❌ | neutral | positive |
| "the movie is a blast of educational energy" | very positive | very positive | very positive |

The clearest failure mode shows up on sentence 3 and 4: TF-IDF has no way to know that "invigorating, electric" is positive if those exact words weren't strongly weighted toward positive in training, and it completely misses sarcasm-adjacent or backhanded phrasing ("a showcase for [X]'s many talents" reads negative to a bag-of-words model because it can't use word order). BERT and the BiLSTM both get closer because they can use context and word order rather than treating the sentence as an unordered bag of features.

**Latency and size** were also benchmarked directly (`outputs/models/`, timed on the same hardware): TF-IDF and the BiLSTM are both sub-millisecond per sample and small enough to fit almost anywhere; BERT is ~5x slower per inference and ~30–150x larger on disk. If latency or edge deployment mattered more than raw accuracy, TF-IDF or the BiLSTM would be the right call — since accuracy was the priority here, BERT is what's wired up behind the actual API.

## What's in the repo

```
sentiment_forge/
├── notebooks/
│   ├── 01_eda.ipynb              # class balance, length distributions, vocab analysis
│   ├── 02_classical.ipynb        # TF-IDF + LogReg, grid search, feature importance
│   ├── 03_BiLSTM.ipynb           # BiLSTM + GloVe, weighted loss, training curves
│   ├── 04_BERT-Fine-tuning.ipynb # BERT fine-tune, W&B tracking, ONNX export
│   └── 05_comparison.ipynb       # head-to-head eval, latency bench, unified error analysis
├── src/
│   ├── data_utils.py    # load / clean / split / persist
│   ├── features.py      # TF-IDF feature union + top-feature inspection
│   ├── models.py        # BiLSTM architecture
│   ├── evaluate.py      # shared metrics, confusion matrices, error analysis
│   └── serve.py         # FastAPI inference app (ONNX Runtime)
├── tests/
│   └── test_pipeline.py # cleaning + metrics unit tests
├── .github/workflows/   # CI: pytest on every push
├── configs/config.yaml  # every hyperparameter, in one place
├── outputs/
│   ├── figures/          # confusion matrices, distributions, training curves
│   ├── models/           # trained models (DVC-tracked, not committed directly)
│   └── reports/          # auto-generated EDA HTML report
└── data/                 # train/val/test parquet (DVC-tracked)
```

## Try it

```bash
pip install -r requirements.txt
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "this movie was absolutely wonderful and touching"}'
```

```json
{
  "text": "this movie was absolutely wonderful and touching",
  "label": "very positive",
  "label_id": 4,
  "confidence": 0.955
}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "terrible film, complete waste of time"}'
# → {"label": "very negative", "confidence": 0.918, ...}
```

Run the tests (7 unit tests covering the cleaning pipeline and metrics functions):

```bash
pytest tests/ -v
```

Reproduce a training run: open any of the numbered notebooks — each one loads its config from `configs/config.yaml`, pulls data via DVC, and logs to MLflow/W&B.

## The MLOps pieces (not just the models)

- **DVC** — datasets and trained model binaries (`.pkl`, `.pt`, ONNX exports) are version-controlled without bloating git; `.dvc` pointer files are what's actually committed
- **MLflow** — every model's metrics logged per run, so results are comparable across experiments rather than just terminal output
- **Weights & Biases** — training curves (loss, val F1 per epoch) tracked for the BiLSTM and BERT runs
- **ONNX export + verification** — BERT was exported to ONNX and the output was numerically checked against the original PyTorch model (max absolute difference: 0.0008) before it went anywhere near the API
- **FastAPI + ONNX Runtime serving** — the API loads the ONNX graph, not the raw PyTorch checkpoint, for faster and lighter inference
- **pytest + GitHub Actions** — 7 unit tests, CI runs automatically on every push to this folder
- **Config-driven** — every hyperparameter (TF-IDF settings, LSTM dims, BERT training args, serving config) lives in one `config.yaml`, not scattered across notebooks

## A few things I'd improve with more time

- Try DistilBERT or a smaller transformer to see how much of BERT's accuracy gain survives at a fraction of the latency/size cost
- Focal loss or a two-stage classifier for the neutral class specifically, since that's the consistent weak point across all three models
- Quantize the ONNX model (int8) to cut inference latency further before deploying it anywhere real
- Push the BiLSTM further — it was only trained for 10 epochs and best-checkpointed at epoch 7, so there may be headroom with a learning rate schedule
