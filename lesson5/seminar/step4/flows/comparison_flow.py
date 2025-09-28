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
    data_manager = ActiveLearningDataManager(random_seed=random_seed)
    data_manager.load_and_split_data()
    return data_manager


@task
def train_baseline_model(
    data_manager: ActiveLearningDataManager, random_seed: int = 42
) -> Dict[str, Any]:
    """Train baseline model on full dataset."""
    baseline_trainer = BaselineTrainer(random_seed=random_seed)
    training_info = baseline_trainer.train_full_dataset(
        X_train=data_manager.X_train_full,
        y_train=data_manager.y_train_full,
        X_val=data_manager.X_val,
        y_val=data_manager.y_val,
        verbose=False,
    )
    val_metrics = baseline_trainer.evaluate_model(
        data_manager.X_val, data_manager.y_val, "validation"
    )
    test_metrics = baseline_trainer.evaluate_model(
        data_manager.X_test, data_manager.y_test, "test"
    )
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
    from src.active_learning import ActiveLearningManager

    al_manager = ActiveLearningManager(uncertainty_strategy=sampling_strategy)
    data_manager.initialize_active_learning(initial_percentage=0.1)
    iteration_results = []

    for iteration in range(max_iterations):
        X_labeled, y_labeled = data_manager.get_labeled_data()
        model_trainer = ModelTrainer(model_params={"random_seed": random_seed})
        model_trainer.train_model(X_labeled, y_labeled)
        training_info = {
            "training_samples": len(X_labeled),
            "features": X_labeled.shape[1],
        }
        val_metrics = model_trainer.evaluate_model(
            data_manager.X_val, data_manager.y_val, "validation"
        )
        test_metrics = model_trainer.evaluate_model(
            data_manager.X_test, data_manager.y_test, "test"
        )
        if data_manager.is_pool_empty():
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

        batch_size = min(
            int(0.1 * len(data_manager.X_train_full)), len(data_manager.X_pool)
        )
        if batch_size <= 0:
            break
        X_pool, y_pool = data_manager.get_pool_data()
        selected_X, selected_y, selected_indices = al_manager.select_next_batch(
            X_pool, y_pool, model_trainer.model, batch_size
        )
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
    mlflow_tracker = MLflowTracker(experiment_name=experiment_name)
    with mlflow_tracker.start_run(run_name="baseline_full_dataset"):
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

        mlflow_tracker.log_metrics(
            {f"val_{k}": v for k, v in baseline_results["val_metrics"].items()}
        )
        mlflow_tracker.log_metrics(
            {f"test_{k}": v for k, v in baseline_results["test_metrics"].items()}
        )
        mlflow_tracker.log_params(baseline_results["training_info"])
    sampling_strategy = al_results["sampling_strategy"]
    with mlflow_tracker.start_run(run_name=f"active_learning_{sampling_strategy}"):
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

        mlflow_tracker.log_metrics(
            {f"val_{k}": v for k, v in final_results["val_metrics"].items()}
        )
        mlflow_tracker.log_metrics(
            {f"test_{k}": v for k, v in final_results["test_metrics"].items()}
        )
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

    data_manager = setup_data_manager(random_seed=random_seed)
    baseline_results = train_baseline_model(data_manager, random_seed=random_seed)
    al_results = {}
    for strategy in sampling_strategies:
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

    for strategy, al_result in al_results.items():
        log_comparison_to_mlflow(
            baseline_results=baseline_results,
            al_results=al_result,
            experiment_name=experiment_name,
        )
    comparison_summary = {
        "baseline": baseline_results,
        "active_learning": al_results,
        "summary": {
            "baseline_test_accuracy": baseline_results["test_metrics"]["accuracy"],
            "baseline_data_usage": "100%",
        },
    }

    for strategy, al_result in al_results.items():
        final_result = al_result["iteration_results"][-1]
        comparison_summary["summary"][f"{strategy}_test_accuracy"] = final_result[
            "test_metrics"
        ]["accuracy"]
        comparison_summary["summary"][
            f"{strategy}_data_usage"
        ] = f"{al_result['final_labeled_samples'] / baseline_results['training_info']['training_samples'] * 100:.1f}%"

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
