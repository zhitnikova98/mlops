.PHONY: install lint test train mlflow-ui docker-build-dev docker-build-prod docker-build docker-run docker-run-dev docker-test docker-shell docker-clean compose-up compose-up-training compose-up-dev compose-up-mlflow compose-up-jupyter compose-down compose-logs compose-clean

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

docker-build-dev:
	@echo "Building Docker development image..."
	docker build --target development -t lesson1-mlops:dev .

docker-build-prod:
	@echo "Building Docker production image..."
	docker build --target production -t lesson1-mlops:0.1 .

docker-build: docker-build-prod

docker-run:
	@echo "Running Docker container..."
	docker run --rm -v "$(PWD)/mlruns:/app/mlruns" lesson1-mlops:0.1

docker-run-dev:
	@echo "Running Docker development container..."
	docker run --rm -it -v "$(PWD)/mlruns:/app/mlruns" -v "$(PWD)/src:/app/src" lesson1-mlops:dev

docker-test:
	@echo "Running tests in Docker..."
	docker run --rm lesson1-mlops:dev poetry run pytest

docker-shell:
	@echo "Opening shell in Docker container..."
	docker run --rm -it --entrypoint /bin/bash lesson1-mlops:dev

docker-clean:
	@echo "Cleaning Docker images..."
	docker rmi lesson1-mlops:dev lesson1-mlops:0.1 || true
	docker system prune -f

# Docker Compose commands
compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up --build

compose-up-training:
	@echo "Running ML training with docker-compose..."
	docker-compose up --build ml-training

compose-up-dev:
	@echo "Starting development environment..."
	docker-compose up --build ml-dev

compose-up-mlflow:
	@echo "Starting MLflow UI..."
	docker-compose up --build mlflow-ui

compose-up-jupyter:
	@echo "Starting Jupyter Lab..."
	docker-compose up --build jupyter

compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

compose-logs:
	@echo "Showing docker-compose logs..."
	docker-compose logs -f

compose-clean:
	@echo "Cleaning docker-compose..."
	docker-compose down --volumes --remove-orphans
	docker-compose rm -f
