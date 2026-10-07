import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import add_domain_features


REASON_MAP = {
    "EXT_SOURCE_1": "Low external credit score rating (Source 1)",
    "EXT_SOURCE_2": "Low external credit bureau score (Source 2)",
    "EXT_SOURCE_3": "Low external bureau risk evaluation (Source 3)",
    "CREDIT_INCOME_RATIO": "High loan amount requested relative to annual income",
    "ANNUITY_INCOME_RATIO": "High monthly loan installment relative to income",
    "ANNUITY_CREDIT_RATIO": "Short loan tenure causing elevated repayment pressure",
    "GOODS_CREDIT_RATIO": "Financing amount exceeds purchase goods value",
    "AMT_CREDIT": "Large total credit obligation requested",
    "AMT_ANNUITY": "High monthly annuity obligation",
    "DAYS_BIRTH": "Younger age credit applicant profile",
    "AGE_YEARS": "Younger age credit profile with limited historical track record",
    "DAYS_EMPLOYED": "Shorter length of current employment tenure",
    "DAYS_ID_PUBLISH": "Recent personal identification document update",
    "REGION_RATING_CLIENT": "Client resides in higher historical credit risk region",
    "REGION_RATING_CLIENT_W_CITY": "City region exhibits higher default incidence",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Historical delinquency events in applicant peer circle (30 DPD)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Historical delinquency events in applicant peer circle (60 DPD)",
    "DAYS_LAST_PHONE_CHANGE": "Recent mobile phone or contact info modification",
    "NAME_EDUCATION_TYPE": "Education category associated with variable income stability",
    "NAME_INCOME_TYPE": "Income stream type carries higher volatility"
}


class PredictPipeline:
    def __init__(
        self,
        preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl"),
        model_path: str = os.path.join("artifacts", "model.pkl"),
        threshold_path: str = os.path.join("artifacts", "threshold.json")
    ):
        try:
            self.preprocessor = load_object(preprocessor_path)
            self.model = load_object(model_path)
            self.booster = self.model.get_booster()

            if os.path.exists(threshold_path):
                with open(threshold_path, "r") as f:
                    t_data = json.load(f)
                    self.threshold = float(t_data.get("threshold", 0.14))
            else:
                self.threshold = 0.14

            self.expected_features = list(self.preprocessor.feature_names_in_)
            self.output_feature_names = [
                f.replace("num_pipeline__", "").replace("cat_pipeline__", "")
                for f in self.preprocessor.get_feature_names_out()
            ]

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, raw_input: dict or pd.DataFrame) -> dict:
        """Takes raw input dictionary or DataFrame and returns calibrated default probability,
        decision (Approved/Rejected), risk band, and top 3 SHAP risk drivers."""
        try:
            if isinstance(raw_input, dict):
                df = pd.DataFrame([raw_input])
            else:
                df = raw_input.copy()

            # Apply domain financial ratio engineering first
            df = add_domain_features(df)

            # Reindex to match expected schema exactly, filling missing columns with NaN
            df_aligned = df.reindex(columns=self.expected_features, fill_value=np.nan)

            # Transform features
            X_transformed = self.preprocessor.transform(df_aligned)

            # Predict probability of default (PD)
            prob = float(self.model.predict_proba(X_transformed)[0, 1])
            prob_percent = round(prob * 100, 2)

            # Decision based on validation-tuned cutoff
            decision = "Approved" if prob < self.threshold else "Rejected"

            # Assign Credit Risk Band (AAA to HR)
            if prob < 0.03:
                risk_band = "Low Risk (Grade A)"
            elif prob < 0.08:
                risk_band = "Moderate Risk (Grade B)"
            elif prob < self.threshold:
                risk_band = "Acceptable Risk (Grade C)"
            elif prob < 0.25:
                risk_band = "Elevated Risk (Grade D)"
            else:
                risk_band = "High Risk (Grade E - Subprime)"

            # Exact Tree SHAP attribution values via XGBoost C++ engine
            dmat = xgb.DMatrix(X_transformed)
            shap_contribs = self.booster.predict(dmat, pred_contribs=True)[0]
            # shap_contribs has shape (num_features + 1,), last element is bias
            feature_shaps = shap_contribs[:-1]

            # Rank features by positive SHAP impact (features pushing towards default)
            top_indices = np.argsort(feature_shaps)[::-1]
            top_reasons = []

            for idx in top_indices:
                feat_name = self.output_feature_names[idx]
                shap_val = float(feature_shaps[idx])
                if shap_val <= 0:
                    continue  # Only consider factors increasing default risk

                # Match base feature
                base_feat = feat_name.split("_")[0] + "_" + feat_name.split("_")[1] if "_" in feat_name else feat_name
                explanation = REASON_MAP.get(feat_name, None)
                if not explanation:
                    for key, val in REASON_MAP.items():
                        if feat_name.startswith(key):
                            explanation = val
                            break
                if not explanation:
                    explanation = f"Risk factor elevation related to {feat_name}"

                top_reasons.append({
                    "feature": feat_name,
                    "shap_importance": round(shap_val, 4),
                    "reason": explanation
                })

                if len(top_reasons) >= 3:
                    break

            # Fallback if no positive SHAP factors (very low risk applicant)
            if not top_reasons:
                top_reasons.append({
                    "feature": "OVERALL_PROFILE",
                    "shap_importance": 0.0,
                    "reason": "Clean credit and employment risk profile"
                })

            return {
                "probability_of_default": round(prob, 4),
                "probability_percentage": f"{prob_percent}%",
                "decision": decision,
                "cutoff_threshold": self.threshold,
                "risk_band": risk_band,
                "top_risk_reasons": top_reasons
            }

        except Exception as e:
            raise CustomException(e, sys)
