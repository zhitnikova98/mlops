"""Prefect flow for comparing Active Learning vs Baseline (Full Dataset)."""

import logging
from prefect import flow, task
from typing import Dict, Any

from src.data_manager import ActiveLearningDataManager
from src.model_trainer import ModelTrainer
from src.baseline_trainer import BaselineTrainer
from src.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


@task
def setup_data_manager(random_seed: int = 42) -> ActiveLearningDataManager:
    """Initialize and load data manager."""
    logger.info("Setting up data manager...")
    data_manager = ActiveLearningDataManager(random_seed=random_seed)
    data_info = data_manager.load_and_split_data()
    logger.info(f"Data loaded: {data_info}")
    return data_manager


@task
def train_baseline_model(
    data_manager: ActiveLearningDataManager, random_seed: int = 42
) -> Dict[str, Any]:
    """Train baseline model on full dataset."""
    logger.info("Training baseline model on full dataset...")

    # Initialize baseline trainer
    baseline_trainer = BaselineTrainer(random_seed=random_seed)

    # Train on full dataset
    training_info = baseline_trainer.train_full_dataset(
        X_train=data_manager.X_train_full,
        y_train=data_manager.y_train_full,
        X_val=data_manager.X_val,
        y_val=data_manager.y_val,
        verbose=False,
    )

    # Evaluate on validation set
    val_metrics = baseline_trainer.evaluate_model(
        data_manager.X_val, data_manager.y_val, "validation"
    )

    # Evaluate on test set
    test_metrics = baseline_trainer.evaluate_model(
        data_manager.X_test, data_manager.y_test, "test"
    )

    # Save model
    baseline_trainer.save_model("models/baseline_full_dataset.pkl")

    return {
        "training_info": training_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_info": baseline_trainer.get_model_info(),
    }


@task
def run_active_learning_experiment(
    data_manager: ActiveLearningDataManager,
    sampling_strategy: str = "entropy",
    max_iterations: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run Active Learning experiment."""
    logger.info(f"Running Active Learning with {sampling_strategy} sampling...")

    from src.active_learning import ActiveLearningManager

    # Initialize Active Learning
    al_manager = ActiveLearningManager(uncertainty_strategy=sampling_strategy)

    # Initialize with 10% of training data
    init_info = data_manager.initialize_active_learning(initial_percentage=0.1)
    logger.info(f"AL initialized: {init_info}")

    # Track results across iterations
    iteration_results = []

    for iteration in range(max_iterations):
        logger.info(f"Active Learning iteration {iteration + 1}/{max_iterations}")

        # Train model on current labeled set
        X_labeled, y_labeled = data_manager.get_labeled_data()
        model_trainer = ModelTrainer(model_params={"random_seed": random_seed})

        model_trainer.train_model(X_labeled, y_labeled)
        training_info = {
            "training_samples": len(X_labeled),
            "features": X_labeled.shape[1],
        }

        # Evaluate model
        val_metrics = model_trainer.evaluate_model(
            data_manager.X_val, data_manager.y_val, "validation"
        )
        test_metrics = model_trainer.evaluate_model(
            data_manager.X_test, data_manager.y_test, "test"
        )

        # Check if pool is empty
        if data_manager.is_pool_empty():
            logger.info("Pool is empty, stopping Active Learning")
            iteration_results.append(
                {
                    "iteration": iteration + 1,
                    "labeled_samples": len(X_labeled),
                    "pool_samples": 0,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "training_info": training_info,
                }
            )
            break

        # Select next batch using uncertainty sampling
        batch_size = min(
            int(0.1 * len(data_manager.X_train_full)), len(data_manager.X_pool)
        )
        if batch_size <= 0:
            logger.info("No more samples to select, stopping")
            break

        X_pool, y_pool = data_manager.get_pool_data()
        selected_X, selected_y, selected_indices = al_manager.select_next_batch(
            X_pool, y_pool, model_trainer.model, batch_size
        )

        # Add selected samples to labeled set
        update_info = data_manager.add_samples_to_labeled_set(selected_indices)

        iteration_results.append(
            {
                "iteration": iteration + 1,
                "labeled_samples": len(X_labeled),
                "pool_samples": len(data_manager.X_pool),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "training_info": training_info,
                "update_info": update_info,
            }
        )

        logger.info(f"Iteration {iteration + 1} completed:")
        logger.info(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Labeled samples: {len(X_labeled)}")

    return {
        "sampling_strategy": sampling_strategy,
        "total_iterations": len(iteration_results),
        "final_labeled_samples": (
            iteration_results[-1]["labeled_samples"] if iteration_results else 0
        ),
        "iteration_results": iteration_results,
    }


@task
def log_comparison_to_mlflow(
    baseline_results: Dict[str, Any],
    al_results: Dict[str, Any],
    experiment_name: str = "step4_comparison",
) -> None:
    """Log comparison results to MLflow."""
    logger.info("Logging comparison results to MLflow...")

    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(experiment_name=experiment_name)

    # Log baseline results
    with mlflow_tracker.start_run(run_name="baseline_full_dataset"):
        # Log parameters
        mlflow_tracker.log_params(
            {
                "method": "baseline",
                "training_samples": baseline_results["training_info"][
                    "training_samples"
                ],
                "data_usage": "100%",
                "model_type": "CatBoost",
            }
        )

        # Log metrics
        mlflow_tracker.log_metrics(
            {f"val_{k}": v for k, v in baseline_results["val_metrics"].items()}
        )
        mlflow_tracker.log_metrics(
            {f"test_{k}": v for k, v in baseline_results["test_metrics"].items()}
        )

        # Log model info
        mlflow_tracker.log_params(baseline_results["training_info"])

    # Log Active Learning results
    sampling_strategy = al_results["sampling_strategy"]
    with mlflow_tracker.start_run(run_name=f"active_learning_{sampling_strategy}"):
        # Log parameters
        final_results = al_results["iteration_results"][-1]

        mlflow_tracker.log_params(
            {
                "method": "active_learning",
                "sampling_strategy": sampling_strategy,
                "total_iterations": al_results["total_iterations"],
                "final_labeled_samples": al_results["final_labeled_samples"],
                "data_usage_percent": (
                    al_results["final_labeled_samples"]
                    / baseline_results["training_info"]["training_samples"]
                    * 100
                ),
            }
        )

        # Log final metrics
        mlflow_tracker.log_metrics(
            {f"val_{k}": v for k, v in final_results["val_metrics"].items()}
        )
        mlflow_tracker.log_metrics(
            {f"test_{k}": v for k, v in final_results["test_metrics"].items()}
        )

        # Log iteration metrics
        for i, result in enumerate(al_results["iteration_results"]):
            mlflow_tracker.log_metrics(
                {
                    f"iteration_{i+1}_val_accuracy": result["val_metrics"]["accuracy"],
                    f"iteration_{i+1}_test_accuracy": result["test_metrics"][
                        "accuracy"
                    ],
                    f"iteration_{i+1}_labeled_samples": result["labeled_samples"],
                },
                step=i + 1,
            )

    logger.info("Comparison results logged to MLflow successfully")


@flow(name="Comparison Pipeline")
def comparison_pipeline(
    sampling_strategies: list = None,
    max_iterations: int = 10,
    random_seed: int = 42,
    experiment_name: str = "step4_comparison",
) -> Dict[str, Any]:
    """Complete comparison pipeline: Active Learning vs Baseline.

    Args:
        sampling_strategies: List of AL sampling strategies to compare
        max_iterations: Maximum AL iterations
        random_seed: Random seed for reproducibility
        experiment_name: MLflow experiment name

    Returns:
        Dictionary with all comparison results
    """
    if sampling_strategies is None:
        sampling_strategies = ["entropy"]

    logger.info("Starting comparison pipeline...")
    logger.info(f"Sampling strategies: {sampling_strategies}")
    logger.info(f"Max iterations: {max_iterations}")

    # Setup data
    data_manager = setup_data_manager(random_seed=random_seed)

    # Train baseline model
    baseline_results = train_baseline_model(data_manager, random_seed=random_seed)

    # Run Active Learning experiments
    al_results = {}
    for strategy in sampling_strategies:
        # Reset Active Learning state for each strategy
        data_manager.X_labeled = None
        data_manager.y_labeled = None
        data_manager.X_pool = None
        data_manager.y_pool = None
        data_manager.labeled_indices = []
        data_manager.pool_indices = []

        al_results[strategy] = run_active_learning_experiment(
            data_manager=data_manager,
            sampling_strategy=strategy,
            max_iterations=max_iterations,
            random_seed=random_seed,
        )

    # Log all results to MLflow
    for strategy, al_result in al_results.items():
        log_comparison_to_mlflow(
            baseline_results=baseline_results,
            al_results=al_result,
            experiment_name=experiment_name,
        )

    # Compile final comparison
    comparison_summary = {
        "baseline": baseline_results,
        "active_learning": al_results,
        "summary": {
            "baseline_test_accuracy": baseline_results["test_metrics"]["accuracy"],
            "baseline_data_usage": "100%",
        },
    }

    # Add AL summaries
    for strategy, al_result in al_results.items():
        final_result = al_result["iteration_results"][-1]
        comparison_summary["summary"][f"{strategy}_test_accuracy"] = final_result[
            "test_metrics"
        ]["accuracy"]
        comparison_summary["summary"][
            f"{strategy}_data_usage"
        ] = f"{al_result['final_labeled_samples'] / baseline_results['training_info']['training_samples'] * 100:.1f}%"

    logger.info("Comparison pipeline completed successfully!")
    logger.info("Summary:")
    for key, value in comparison_summary["summary"].items():
        logger.info(f"  {key}: {value}")

    return comparison_summary


@flow(name="Single Strategy Comparison")
def single_strategy_comparison(
    sampling_strategy: str = "entropy",
    max_iterations: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run comparison for a single AL strategy vs baseline."""
    return comparison_pipeline(
        sampling_strategies=[sampling_strategy],
        max_iterations=max_iterations,
        random_seed=random_seed,
        experiment_name=f"step4_single_{sampling_strategy}",
    )
