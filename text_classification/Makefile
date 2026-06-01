# ── Project 01 · Text Classification Pipeline ─────────────────────────────
# Usage:
#   make install    → install all dependencies
#   make data       → download + validate + version raw data
#   make profile    → generate ydata-profiling EDA report
#   make train      → run all three training notebooks sequentially
#   make serve      → launch FastAPI serving endpoint
#   make test       → run pytest suite
#   make lint       → ruff linting + formatting check
#   make clean      → remove generated artefacts (not data)

.PHONY: install data profile train serve test lint clean

# ── Python & paths ─────────────────────────────────────────────────────────
PYTHON     := python3
PIP        := pip
SRC        := src
NOTEBOOKS  := notebooks
CONFIGS    := configs/config.yaml

# ── Install ────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m pip install -e .
	@echo "✓ dependencies installed"

# ── Data pipeline ──────────────────────────────────────────────────────────
data:
	$(PYTHON) -c "from src.data_utils import download_and_validate; download_and_validate('$(CONFIGS)')"
	dvc add data/raw
	dvc push
	@echo "✓ raw data downloaded, validated, and versioned with DVC"

# ── EDA profiling ──────────────────────────────────────────────────────────
profile:
	$(PYTHON) -c "from src.data_utils import run_profiling; run_profiling('$(CONFIGS)')"
	@echo "✓ profiling report saved to data/profiles/"

# ── Training ───────────────────────────────────────────────────────────────
train: train-classical train-neural train-transformer

train-classical:
	jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=600 \
		$(NOTEBOOKS)/02_classical_pipeline.ipynb \
		--output $(NOTEBOOKS)/02_classical_pipeline_executed.ipynb
	@echo "✓ classical pipeline trained"

train-neural:
	jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=1800 \
		$(NOTEBOOKS)/03_neural_pipeline.ipynb \
		--output $(NOTEBOOKS)/03_neural_pipeline_executed.ipynb
	@echo "✓ neural pipeline trained"

train-transformer:
	jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=3600 \
		$(NOTEBOOKS)/04_transformer_pipeline.ipynb \
		--output $(NOTEBOOKS)/04_transformer_pipeline_executed.ipynb
	@echo "✓ transformer pipeline trained"

# ── Serving ────────────────────────────────────────────────────────────────
serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
	@echo "✓ API running at http://localhost:8000"

# ── Testing ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short
	@echo "✓ all tests passed"

# ── Linting ────────────────────────────────────────────────────────────────
lint:
	ruff check $(SRC) tests/
	ruff format --check $(SRC) tests/
	@echo "✓ linting passed"

format:
	ruff format $(SRC) tests/
	@echo "✓ code formatted"

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf mlruns/ logs/ models/
	@echo "✓ artefacts cleaned (data preserved)"
