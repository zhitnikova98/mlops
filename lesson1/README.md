# Lesson 1: Reproducible ML Pipeline with MLflow

This project demonstrates a minimal reproducible ML pipeline using:
- Logistic Regression on Iris dataset
- MLflow tracking for experiments
- Docker containerization
- Code quality tools (black, ruff, mypy)
- Unit tests with pytest

## Project Structure

```
lesson1/
├── configs/
│   └── train.yaml          # Training configuration
├── src/app/
│   ├── __init__.py
│   └── train.py            # Main training script
├── tests/
│   └── test_sanity.py      # Unit tests
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── pyproject.toml          # Poetry dependencies
├── Makefile               # Common commands
├── Dockerfile             # Container definition
├── .dockerignore
├── .gitignore
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.11
- Poetry (recommended) or pip
- Docker (optional)

### Install Dependencies
```bash
make install
```

## Usage

### Training
```bash
make train
```

Expected metrics:
- Accuracy: ~0.9-1.0
- F1 Score (macro): ~0.9-1.0

### Running Tests
```bash
make test
```

### Code Quality
```bash
make lint
```

### MLflow UI
```bash
make mlflow-ui
```
Then open http://localhost:5000

### Docker
```bash
make docker-build
make docker-run
```

## MLflow Artifacts

After training, MLflow artifacts are stored in `./mlruns/` directory:
- Parameters: seed, test_size, C, max_iter
- Metrics: accuracy, f1_macro
- Model: sklearn LogisticRegression
- Artifacts: train.yaml config

## Reproducibility

The pipeline uses fixed seeds (42) for:
- Random state
- NumPy random state
- Sklearn train_test_split
- Model initialization

Running `make train` twice should produce identical results (within 0.001 tolerance).
