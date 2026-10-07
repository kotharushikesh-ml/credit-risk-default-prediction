import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_pipeline():
    try:
        logging.info("Training pipeline execution started")

        # Step 1: Data Ingestion (Stratified 70/15/15 split)
        logging.info("=== STEP 1: Data Ingestion ===")
        data_ingestion = DataIngestion()
        train_path, val_path, test_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation (Domain financial ratios, ID drop, Preprocessor fit)
        logging.info("=== STEP 2: Data Transformation ===")
        data_transformation = DataTransformation()
        train_arr, val_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path
            )
        )

        # Step 3: Model Training (Baseline vs XGBoost, Validation Threshold, Evaluation)
        logging.info("=== STEP 3: Model Training & Evaluation ===")
        model_trainer = ModelTrainer()
        results = model_trainer.initiate_model_trainer(
            train_array=train_arr,
            val_array=val_arr,
            test_array=test_arr
        )

        base_res = results["baseline_metrics"]
        final_res = results["final_metrics"]

        print("\n" + "=" * 60)
        print("CREDIT RISK DEFAULT MODEL - PIPELINE RUN SUMMARY")
        print("=" * 60)
        print(f"Optimal Decision Threshold (Validation F1): {results['threshold']}")
        print("-" * 60)
        print(f"{'Metric':<20} | {'Baseline (Logistic Reg)':<25} | {'Final (XGBoost)':<20}")
        print("-" * 60)
        print(f"{'ROC AUC':<20} | {base_res['roc_auc']:<25.4f} | {final_res['roc_auc']:<20.4f}")
        print(f"{'Gini (2*AUC - 1)':<20} | {base_res['gini']:<25.4f} | {final_res['gini']:<20.4f}")
        print(f"{'KS Statistic':<20} | {base_res['ks_statistic']:<25.4f} | {final_res['ks_statistic']:<20.4f}")
        print(f"{'Brier Score':<20} | {base_res['brier_score']:<25.4f} | {final_res['brier_score']:<20.4f}")
        print(f"{'Precision':<20} | {base_res['precision']:<25.4f} | {final_res['precision']:<20.4f}")
        print(f"{'Recall':<20} | {base_res['recall']:<25.4f} | {final_res['recall']:<20.4f}")
        print(f"{'F1 Score':<20} | {base_res['f1']:<25.4f} | {final_res['f1']:<20.4f}")
        print("=" * 60)

        logging.info("Training pipeline execution completed successfully")
        return results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()