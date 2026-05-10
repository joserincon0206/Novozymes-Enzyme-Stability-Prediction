# Novozymes Enzyme Stability Prediction

Protein thermostability prediction from amino acid sequences using PyTorch Lightning, MLflow, Optuna, and HuggingFace Transformers (ESM-2).

**Competition**: [Kaggle Novozymes 2022](https://www.kaggle.com/c/novozymes-enzyme-stability-prediction)  
**Evaluation Metric**: Spearman rank correlation  
**Dataset**: Protein sequences + pH → thermostability (tm)

---

## Setup

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repo_url>
cd Novozymes-Enzyme-Stability-Prediction

# Install dependencies and create venv
uv sync --all-extras

# Configure environment
cp .env.example .env

# Set up pre-commit hooks
uv run pre-commit install

# Download data
# Place train.csv and test.csv in data/raw/
```

---

## Architecture

```
src/novozymes/
├── config/                    # Pydantic BaseModel configs (frozen, validated)
│   ├── base.py               # BaseConfig
│   ├── data.py               # DataConfig
│   ├── model.py              # ModelConfig, FineTuneModelConfig
│   └── training.py           # TrainingConfig, LossConfig, OptunaConfig
│
├── data/                      # Data loading and preprocessing
│   ├── dataset.py            # EnzymeDataset (torch.Dataset)
│   ├── datamodule.py         # EnzymeDataModule (LightningDataModule)
│   └── transforms.py         # Sequence cleaning, normalization
│
├── features/                  # Feature extraction
│   ├── base.py               # FeatureExtractor ABC
│   ├── esm_embedding.py      # ESM-2 embeddings + pooling
│   └── esm_tokenizer.py      # Sequence → token tensors
│
├── models/                    # Model strategies
│   ├── base.py               # ModelStrategy ABC
│   ├── registry.py           # STRATEGY_REGISTRY + build_strategy()
│   ├── esm_frozen.py         # Strategy A: frozen ESM-2 + head
│   ├── esm_finetune.py       # Strategy B: fine-tuned ESM-2 + head
│   └── heads/
│       ├── base.py           # Head ABC
│       ├── regression.py     # RegressionHead ([B, 1])
│       ├── coral.py          # CORALHead ([B, K])
│       ├── cogol.py          # CoGOLHead ([B, K])
│       └── registry.py       # HEAD_REGISTRY + build_head()
│
├── training/                  # Training orchestration
│   ├── lightning_module.py    # EnzymeLightningModule
│   └── losses/
│       ├── base.py           # LossStrategy ABC
│       ├── mse.py            # Scalar regression loss
│       ├── coral.py          # Ordinal regression (CORAL)
│       ├── cogol.py          # Ordinal regression (CoGOL)
│       └── registry.py       # LOSS_REGISTRY + build_loss()
│
├── evaluation/                # Metrics
│   └── metrics.py            # Evaluator ABC, SpearmanEvaluator
│
├── exploration/               # Data analysis (pre-training)
│   ├── label_stats.py        # Label distribution analysis
│   ├── bin_advisor.py        # Bin recommendations (CORAL/CoGOL)
│   ├── group_balance.py      # Data source balance
│   ├── sequence_stats.py     # Sequence length, AA frequency
│   └── embedding_viz.py      # PCA/t-SNE visualization
│
├── tuning/                    # Hyperparameter optimization
│   └── optuna_tuner.py       # Optuna + MLflow integration
│
└── utils/
    ├── logging.py            # MLFlowLogger setup
    └── seed.py               # Deterministic seeding

scripts/
├── explore.py               # Data exploration entry point
├── train.py                 # Single training run
├── tune.py                  # Optuna HPO
└── predict.py               # Generate submission.csv

tests/
└── [mirrors src/ structure]  # TDD: test first
```

---

## Quick Start

### 1. Explore Data (Required First Step)

```bash
uv run python scripts/explore.py
```

Outputs:
- Label distribution statistics
- BinAdvisor recommendations (num_classes for ordinal losses)
- Data source balance analysis
- Sequence length and AA frequency distributions

### 2. Start MLflow Server

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Visit `http://localhost:5000` for experiment tracking.

### 3. Train Baseline

```bash
# ESM-2 frozen + MSE loss (baseline)
uv run python scripts/train.py
```

### 4. Test Ordinal Losses

```bash
# Using num_classes from BinAdvisor
uv run python scripts/train.py --loss coral --num_classes 50
uv run python scripts/train.py --loss cogol --num_classes 50
```

### 5. Hyperparameter Tuning

```bash
uv run python scripts/tune.py --n-trials 20 --n-epochs 10
```

### 6. Generate Submission

```bash
uv run python scripts/predict.py \
    --checkpoint mlruns/<exp_id>/<run_id>/artifacts/best.ckpt \
    --output submission.csv
```

---

## Model Strategies

| Strategy | File | Features |
|----------|------|----------|
| `esm_frozen` | `models/esm_frozen.py` | Frozen ESM-2 + trainable head |
| `esm_finetune` | `models/esm_finetune.py` | Fine-tune last N layers + gradient checkpointing |

Add new strategies in 3 files: config, implementation, registry.

---

## Loss Functions

| Loss | File | Head Output | Use Case |
|------|------|-------------|----------|
| `mse` | `training/losses/mse.py` | [B, 1] | Baseline regression |
| `coral` | `training/losses/coral.py` | [B, num_classes] | Ordinal ranking (Consistent Rank Logits) |
| `cogol` | `training/losses/cogol.py` | [B, num_classes] | Ordinal ranking (Generalized Ordinal Loss) |

Each loss declares its required head via `required_head` property. Head selection is registry-based (zero if/else branching).

---

## Configuration

All configs are Pydantic frozen BaseModels. No YAML.

```python
from novozymes.config.data import DataConfig
from novozymes.config.model import ModelConfig
from novozymes.config.training import TrainingConfig, LossConfig

data_cfg = DataConfig(batch_size=32, val_fraction=0.1)
model_cfg = ModelConfig(strategy_name="esm_frozen", hidden_dim=256)
loss_cfg = LossConfig(loss_name="mse")
train_cfg = TrainingConfig(max_epochs=50)
```

---

## Development

### Running Tests (TDD)

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/novozymes --cov-report=html

# Single module
uv run pytest tests/models/ -v
```

### Code Quality

```bash
# Lint + auto-fix
uv run ruff check . --fix

# Format
uv run ruff format .

# Type checking
uv run mypy src/novozymes --strict --ignore-missing-imports

# All checks (run before commit)
uv run pre-commit run --all-files
```

### Pre-commit Hooks

The `.pre-commit-config.yaml` runs on every commit:
- **ruff**: linting and formatting
- **mypy**: static type checking
- **pytest**: unit tests (optional, can be slow)

---

## Extending

### Adding a Model Strategy

1. Create config subclass: `config/model.py`
2. Implement strategy: `models/esm_*.py`
3. Register: one line in `models/registry.py`

### Adding a Loss + Head

1. Implement loss: `training/losses/new_loss.py` (declare `required_head`)
2. Implement head: `models/heads/new_head.py`
3. Register: one line each in `training/losses/registry.py` and `models/heads/registry.py`

---

## Command Reference

```bash
# Setup
uv sync --all-extras              # Install all dependencies
uv run pip list                   # Check installed packages

# Testing
uv run pytest                     # Run all tests
uv run pytest tests/models/ -v   # Run module tests
uv run pytest --cov              # With coverage

# Code quality
uv run ruff check . --fix         # Lint + fix
uv run ruff format .              # Format
uv run mypy src/                  # Type check

# Training
uv run python scripts/explore.py      # Data exploration
uv run python scripts/train.py        # Single training run
uv run python scripts/tune.py         # HPO
uv run python scripts/predict.py      # Submission

# MLflow
uv run mlflow server --host 127.0.0.1 --port 5000
```

---

## Environment Variables (.env)

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
HF_HOME=./cache/huggingface
RANDOM_SEED=42
CUDA_VISIBLE_DEVICES=0
```

---

## References

- [PyTorch Lightning](https://lightning.ai/)
- [MLflow](https://mlflow.org/)
- [Optuna](https://optuna.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Pydantic](https://docs.pydantic.dev/)
- [Ruff](https://docs.astral.sh/ruff/)
- [uv](https://docs.astral.sh/uv/)

---

## License

MIT
