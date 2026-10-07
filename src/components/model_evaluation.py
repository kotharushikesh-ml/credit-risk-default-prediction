import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.exception import CustomException
from src.logger import logging


def compute_credit_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes standard credit risk metrics including AUC, Gini, KS, Brier, and 10-decile table."""
    try:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        y_pred = (y_prob >= threshold).astype(int)

        # 1. Discrimination & Calibration metrics
        auc = float(roc_auc_score(y_true, y_prob))
        gini = float(2 * auc - 1)

        # KS statistic
        prob_bads = y_prob[y_true == 1]
        prob_goods = y_prob[y_true == 0]
        ks_stat = float(ks_2samp(prob_bads, prob_goods).statistic)

        # Brier Score
        brier = float(brier_score_loss(y_true, y_prob))

        # Classification metrics at threshold
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        # 2. 10-Decile Bad Rate Table (Decile 1 = Highest Risk)
        df_eval = pd.DataFrame({"actual": y_true, "prob": y_prob})
        # Rank descending by predicted default probability
        df_eval["decile"] = pd.qcut(df_eval["prob"].rank(method="first", ascending=False), q=10, labels=list(range(1, 11)))

        total_bads = int(df_eval["actual"].sum())
        total_obs = int(len(df_eval))
        overall_bad_rate = float(total_bads / total_obs) if total_obs > 0 else 0.0

        decile_table = []
        cum_bads = 0
        for d in range(1, 11):
            subset = df_eval[df_eval["decile"] == d]
            cnt = int(len(subset))
            bads = int(subset["actual"].sum())
            goods = cnt - bads
            bad_rate = float(bads / cnt) if cnt > 0 else 0.0
            cum_bads += bads
            cum_bad_rate = float(cum_bads / ((d / 10.0) * total_obs)) if total_obs > 0 else 0.0
            bad_capture = float(bads / total_bads) if total_bads > 0 else 0.0
            cum_bad_capture = float(cum_bads / total_bads) if total_bads > 0 else 0.0
            lift = float(bad_rate / overall_bad_rate) if overall_bad_rate > 0 else 0.0

            decile_table.append({
                "decile": d,
                "count": cnt,
                "bads": bads,
                "goods": goods,
                "bad_rate": round(bad_rate, 4),
                "cum_bad_rate": round(cum_bad_rate, 4),
                "bad_capture_rate": round(bad_capture, 4),
                "cum_bad_capture_rate": round(cum_bad_capture, 4),
                "lift": round(lift, 2),
                "min_prob": round(float(subset["prob"].min()), 4) if cnt > 0 else 0.0,
                "max_prob": round(float(subset["prob"].max()), 4) if cnt > 0 else 0.0
            })

        return {
            "roc_auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks_stat, 4),
            "brier_score": round(brier, 4),
            "threshold": round(threshold, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
            "overall_bad_rate": round(overall_bad_rate, 4),
            "total_samples": total_obs,
            "decile_table": decile_table
        }

    except Exception as e:
        raise CustomException(e, sys)


class ModelEvaluation:
    def __init__(self, metrics_output_path: str = os.path.join("artifacts", "metrics.json")):
        self.metrics_output_path = metrics_output_path

    def evaluate_and_save(self, baseline_metrics: dict, final_metrics: dict):
        """Saves evaluation comparison dictionary to artifacts/metrics.json."""
        try:
            os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
            report = {
                "baseline_logistic_regression": baseline_metrics,
                "final_xgboost_model": final_metrics
            }
            with open(self.metrics_output_path, "w") as f:
                json.dump(report, f, indent=4)
            logging.info("Metrics report saved to %s", self.metrics_output_path)
            return report
        except Exception as e:
            raise CustomException(e, sys)
