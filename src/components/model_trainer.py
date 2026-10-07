import json
import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.model_evaluation import compute_credit_metrics, ModelEvaluation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    baseline_model_file_path: str = os.path.join("artifacts", "baseline_model.pkl")
    threshold_file_path: str = os.path.join("artifacts", "threshold.json")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array: np.ndarray,
        val_array: np.ndarray,
        test_array: np.ndarray
    ):
        try:
            logging.info("Splitting feature arrays into X and y for train, val, and test")
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_val, y_val = val_array[:, :-1], val_array[:, -1].astype(int)
            X_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            logging.info("Training features shape: X_train=%s, X_val=%s, X_test=%s",
                         X_train.shape, X_val.shape, X_test.shape)

            # ==========================================
            # 1. Baseline Model: Logistic Regression
            # ==========================================
            logging.info("Training Baseline Model: Logistic Regression")
            baseline_model = LogisticRegression(
                max_iter=500,
                random_state=42,
                solver="lbfgs"
            )
            baseline_model.fit(X_train, y_train)

            # Baseline validation probabilities & default 0.5 threshold
            baseline_val_proba = baseline_model.predict_proba(X_val)[:, 1]
            baseline_test_proba = baseline_model.predict_proba(X_test)[:, 1]

            # Find validation threshold for baseline
            best_base_threshold = 0.5
            best_base_f1 = 0.0
            for t in np.arange(0.05, 0.55, 0.01):
                f1 = f1_score(y_val, (baseline_val_proba >= t).astype(int), zero_division=0)
                if f1 > best_base_f1:
                    best_base_f1 = f1
                    best_base_threshold = float(t)

            baseline_metrics = compute_credit_metrics(
                y_true=y_test,
                y_prob=baseline_test_proba,
                threshold=best_base_threshold
            )
            logging.info("Baseline Logistic Regression Test ROC AUC: %.4f, Gini: %.4f, KS: %.4f",
                         baseline_metrics["roc_auc"], baseline_metrics["gini"], baseline_metrics["ks_statistic"])

            # ==========================================
            # 2. Final Model: XGBoost Classifier
            # (scale_pos_weight removed to ensure calibrated probabilities)
            # ==========================================
            logging.info("Training Final Model: XGBClassifier (calibrated default probabilities)")
            xgb_model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                tree_method="hist"
            )
            xgb_model.fit(X_train, y_train)

            xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]

            # ==========================================
            # 3. Threshold Tuning strictly on Validation set
            # ==========================================
            logging.info("Tuning decision threshold strictly on Validation set")
            best_threshold = 0.5
            best_f1 = 0.0

            for t in np.arange(0.05, 0.55, 0.01):
                pred = (xgb_val_proba >= t).astype(int)
                f1 = f1_score(y_val, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_threshold = float(round(t, 2))

            logging.info("Optimal threshold on validation set: %.2f (Validation F1: %.4f)",
                         best_threshold, best_f1)

            # ==========================================
            # 4. Final Evaluation strictly on Test set
            # ==========================================
            xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
            final_metrics = compute_credit_metrics(
                y_true=y_test,
                y_prob=xgb_test_proba,
                threshold=best_threshold
            )

            logging.info("Final XGBoost Test ROC AUC: %.4f, Gini: %.4f, KS: %.4f, Brier: %.4f, F1: %.4f",
                         final_metrics["roc_auc"], final_metrics["gini"],
                         final_metrics["ks_statistic"], final_metrics["brier_score"],
                         final_metrics["f1"])

            # Save threshold metadata
            os.makedirs(os.path.dirname(self.model_trainer_config.threshold_file_path), exist_ok=True)
            threshold_data = {
                "threshold": best_threshold,
                "validation_f1": round(best_f1, 4),
                "evaluation_metric": "f1_score"
            }
            with open(self.model_trainer_config.threshold_file_path, "w") as f:
                json.dump(threshold_data, f, indent=4)
            logging.info("Saved threshold to %s", self.model_trainer_config.threshold_file_path)

            # Save baseline and final models
            save_object(self.model_trainer_config.trained_model_file_path, xgb_model)
            save_object(self.model_trainer_config.baseline_model_file_path, baseline_model)
            logging.info("Saved models to %s and %s",
                         self.model_trainer_config.trained_model_file_path,
                         self.model_trainer_config.baseline_model_file_path)

            # Save metrics report
            evaluator = ModelEvaluation(self.model_trainer_config.metrics_file_path)
            report = evaluator.evaluate_and_save(baseline_metrics, final_metrics)

            return {
                "threshold": best_threshold,
                "baseline_metrics": baseline_metrics,
                "final_metrics": final_metrics
            }

        except Exception as e:
            raise CustomException(e, sys)