import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 5 standard credit risk financial ratio features."""
    df = df.copy()
    # Ratio 1: Credit to Income
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    # Ratio 2: Annuity to Income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    # Ratio 3: Annuity to Credit
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
    # Ratio 4: Goods Price to Credit
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1e-5)
    # Ratio 5: Age in years
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    return df


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, sample_input_df: pd.DataFrame):
        """Constructs and returns the scikit-learn ColumnTransformer."""
        try:
            numerical_columns = sample_input_df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_columns = sample_input_df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            logging.info("Numerical Columns count: %d", len(numerical_columns))
            logging.info("Categorical Columns count: %d", len(categorical_columns))

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, val_path: str, test_path: str):
        """Transforms train, val, and test splits and saves the preprocessor object."""
        try:
            logging.info("Reading train, validation, and test sets for transformation")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logging.info("Adding domain credit ratio features")
            train_df = add_domain_features(train_df)
            val_df = add_domain_features(val_df)
            test_df = add_domain_features(test_df)

            # Drop identifier and target column to prevent data leakage
            drop_cols = ["TARGET", "SK_ID_CURR"]
            existing_drop_cols = [c for c in drop_cols if c in train_df.columns]

            X_train = train_df.drop(columns=existing_drop_cols, axis=1)
            y_train = train_df["TARGET"].values

            X_val = val_df.drop(columns=existing_drop_cols, axis=1)
            y_val = val_df["TARGET"].values

            X_test = test_df.drop(columns=existing_drop_cols, axis=1)
            y_test = test_df["TARGET"].values

            logging.info("Training feature dimensions: %s", X_train.shape)

            preprocessing_obj = self.get_data_transformer_object(X_train)

            logging.info("Fitting preprocessor on training data and transforming splits")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_val_arr = preprocessing_obj.transform(X_val)
            X_test_arr = preprocessing_obj.transform(X_test)

            logging.info("Transformed feature matrix shape: %s", X_train_arr.shape)

            train_arr = np.c_[X_train_arr, y_train]
            val_arr = np.c_[X_val_arr, y_val]
            test_arr = np.c_[X_test_arr, y_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Saved preprocessor object to %s", self.data_transformation_config.preprocessor_obj_file_path)

            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)