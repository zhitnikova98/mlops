.PHONY: install lint test train mlflow-ui docker-build docker-run

install:
	@echo "Installing dependencies with Poetry..."
	poetry install

lint:
	@echo "Running linting..."
	poetry run ruff check .
	poetry run black --check .

test:
	@echo "Running tests..."
	poetry run pytest -q

train:
	@echo "Running training..."
	poetry run python -m src.app.train

mlflow-ui:
	@echo "Starting MLflow UI..."
	poetry run mlflow ui --backend-store-uri ./mlruns

docker-build:
	@echo "Building Docker image..."
	docker build -t lesson1-mlops:0.1 .

docker-run:
	@echo "Running Docker container..."
	docker run --rm -v "$(PWD)/mlruns:/app/mlruns" -w /app lesson1-mlops:0.1 python -m src.app.train
