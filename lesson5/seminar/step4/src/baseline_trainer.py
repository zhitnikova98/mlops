"""Baseline trainer for full dataset comparison."""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Any
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """Trains model on full dataset for comparison with Active Learning."""

    def __init__(self, random_seed: int = 42):
        """Initialize BaselineTrainer.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.is_trained = False

    def train_full_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train CatBoost model on full training dataset.

        Args:
            X_train: Full training features
            y_train: Full training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to show training progress

        Returns:
            Dictionary with training information
        """
        logger.info("Training baseline model on full dataset...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

        # Initialize CatBoost classifier
        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_seed=self.random_seed,
            verbose=verbose,
            early_stopping_rounds=50,
            eval_metric="Accuracy",
        )

        # Train model with validation set for early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=verbose,
        )

        self.is_trained = True

        # Get training information
        training_info = {
            "model_type": "CatBoost",
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "features": X_train.shape[1],
            "classes": len(np.unique(y_train)),
            "iterations_trained": self.model.tree_count_,
            "best_iteration": self.model.get_best_iteration(),
        }

        logger.info("Baseline training completed:")
        logger.info(f"  Best iteration: {training_info['best_iteration']}")
        logger.info(f"  Total iterations: {training_info['iterations_trained']}")

        return training_info

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray, dataset_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate trained model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of dataset being evaluated

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_full_dataset() first.")

        logger.info(f"Evaluating baseline model on {dataset_name} set...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }

        logger.info(f"Baseline {dataset_name} metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save trained model to file.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_full_dataset() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Baseline model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from file.

        Args:
            filepath: Path to load model from
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info(f"Baseline model loaded from: {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "model_type": "CatBoost",
            "tree_count": self.model.tree_count_,
            "best_iteration": self.model.get_best_iteration(),
            "feature_importances_available": True,
        }

    def get_feature_importance(self, top_k: int = 10) -> Dict[str, float]:
        """Get top-k feature importances.

        Args:
            top_k: Number of top features to return

        Returns:
            Dictionary with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_full_dataset() first.")

        # Get feature importances
        importances = self.model.get_feature_importance()
        feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort by importance and get top-k
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)

        return dict(importance_pairs[:top_k])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_full_dataset() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Features to predict

        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_full_dataset() first.")

        return self.model.predict_proba(X)
