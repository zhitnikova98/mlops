.PHONY: help install setup start-services stop-services prefect-server mlflow-ui run-al run-al-entropy run-al-margin run-al-confident run-baseline compare-all run-single benchmark clean

# Default target
help:  ## Show help message
	@echo "Available commands:"
	@echo "  install         - Install dependencies using Poetry"
	@echo "  setup           - Setup project directories"
	@echo "  start-services  - Start both Prefect and MLflow services"
	@echo "  prefect-server  - Start Prefect server (run in separate terminal)"
	@echo "  mlflow-ui       - Start MLflow UI (run in separate terminal)"
	@echo "  run-al          - Run Active Learning pipeline (entropy strategy)"
	@echo "  run-al-entropy  - Run AL with entropy uncertainty sampling"
	@echo "  run-al-margin   - Run AL with margin uncertainty sampling"
	@echo "  run-al-confident- Run AL with least confident uncertainty sampling"
	@echo "  run-baseline    - Run baseline training on full dataset"
	@echo "  compare-all     - Compare all AL strategies vs baseline"
	@echo "  run-single      - Run single AL iteration"
	@echo "  benchmark       - Run comprehensive benchmark of all methods"
	@echo "  clean           - Clean all generated files and artifacts"
	@echo "  stop-services   - Stop all running services"
	@echo ""
	@echo "Example usage:"
	@echo "  make install && make setup"
	@echo "  make start-services      # Start both services"
	@echo "  make compare-all         # Run full comparison"

# Install dependencies
install:  ## Install dependencies
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed successfully!"

# Setup project directories
setup:  ## Setup project directories
	@echo "Setting up project directories..."
	mkdir -p models metrics mlruns
	@echo "Project setup completed!"

# Start both services
start-services:  ## Start Prefect and MLflow services
	@echo "Starting Prefect server in background..."
	poetry run prefect server start --host 0.0.0.0 > prefect.log 2>&1 &
	@echo "Starting MLflow UI in background..."
	poetry run mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
	@echo "Waiting for services to start..."
	sleep 10
	@echo "Services started!"
	@echo "  Prefect UI: http://localhost:4200"
	@echo "  MLflow UI: http://localhost:5000"

# Start Prefect server
prefect-server:  ## Start Prefect server
	@echo "Starting Prefect server..."
	@echo "Prefect UI will be available at: http://localhost:4200"
	poetry run prefect server start --host 0.0.0.0

# Start MLflow UI
mlflow-ui:  ## Start MLflow UI
	@echo "Starting MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:5000"
	poetry run mlflow ui --host 0.0.0.0 --port 5000

# Run Active Learning pipeline with entropy strategy (default)
run-al:  ## Run Active Learning pipeline
	@echo "Running Active Learning pipeline with entropy strategy..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='entropy', max_iterations=10)"

# Run Active Learning with entropy uncertainty sampling
run-al-entropy:  ## Run AL with entropy sampling
	@echo "Running Active Learning with entropy uncertainty sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='entropy', max_iterations=10)"

# Run Active Learning with margin uncertainty sampling
run-al-margin:  ## Run AL with margin sampling
	@echo "Running Active Learning with margin uncertainty sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='margin', max_iterations=10)"

# Run Active Learning with least confident uncertainty sampling
run-al-confident:  ## Run AL with least confident sampling
	@echo "Running Active Learning with least confident uncertainty sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='least_confident', max_iterations=10)"

# Run baseline training on full dataset
run-baseline:  ## Run baseline training
	@echo "Running baseline training on full dataset..."
	poetry run python -c "from flows.comparison_flow import comparison_pipeline; comparison_pipeline(sampling_strategies=[], max_iterations=0, experiment_name='step4_baseline_only')"

# Compare all AL strategies vs baseline
compare-all:  ## Compare all methods
	@echo "Running comprehensive comparison: AL strategies vs baseline..."
	poetry run python -c "from flows.comparison_flow import comparison_pipeline; comparison_pipeline(sampling_strategies=['entropy', 'margin', 'least_confident'], max_iterations=10, experiment_name='step4_full_comparison')"

# Run single AL iteration for testing
run-single:  ## Run single AL iteration
	@echo "Running single Active Learning iteration..."
	poetry run python -c "from flows.comparison_flow import single_strategy_comparison; single_strategy_comparison(sampling_strategy='entropy', max_iterations=1)"

# Run comprehensive benchmark
benchmark:  ## Run comprehensive benchmark
	@echo "Running comprehensive benchmark of all methods..."
	@echo "This will take several minutes..."
	poetry run python -c "\
from flows.comparison_flow import comparison_pipeline; \
import time; \
print('=== BENCHMARK START ==='); \
start_time = time.time(); \
results = comparison_pipeline(\
    sampling_strategies=['entropy', 'margin', 'least_confident'], \
    max_iterations=10, \
    experiment_name='step4_benchmark'\
); \
end_time = time.time(); \
print(f'=== BENCHMARK COMPLETED in {end_time - start_time:.2f}s ==='); \
print('Results summary:'); \
for key, value in results['summary'].items(): \
    print(f'  {key}: {value}'); \
"

# Clean generated files
clean:  ## Clean generated files
	@echo "Cleaning generated files..."
	rm -rf models/
	rm -rf metrics/
	rm -rf mlruns/
	rm -rf catboost_info/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf flows/__pycache__/
	@echo "Cleanup completed!"

# Stop services
stop-services:  ## Stop all services
	@echo "Stopping services..."
	pkill -f "prefect server" || true
	pkill -f "mlflow ui" || true
	@echo "Services stopped!"
