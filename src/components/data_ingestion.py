import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "application_train.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    val_data_path: str = os.path.join("artifacts", "val.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            logging.info("Reading raw dataset from %s", self.ingestion_config.raw_data_path)
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Dataset loaded successfully. Shape: %s", df.shape)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Initiating Stratified 70/15/15 Train/Validation/Test split")

            # 70% train, 30% temp (stratified on TARGET)
            train_set, temp_set = train_test_split(
                df,
                test_size=0.30,
                random_state=42,
                stratify=df["TARGET"]
            )

            # Split temp into 50% val (15% total), 50% test (15% total) (stratified on TARGET)
            val_set, test_set = train_test_split(
                temp_set,
                test_size=0.50,
                random_state=42,
                stratify=temp_set["TARGET"]
            )

            logging.info("Split completed: Train=%d, Val=%d, Test=%d", len(train_set), len(val_set), len(test_set))
            logging.info("Default rate: Train=%.4f, Val=%.4f, Test=%.4f",
                         train_set["TARGET"].mean(), val_set["TARGET"].mean(), test_set["TARGET"].mean())

            # Save split sets without duplicating full raw data to artifacts/
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion artifacts saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)