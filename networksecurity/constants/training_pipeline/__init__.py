import os
import sys
import numpy as np
import pandas as pd


TARGET_COLUMN:str = "Result"
PIPELINE_NAME:str = "network_security_pipeline"
ARTIFACT_DIR:str = "artifact"
FILE_NAME:str = "phishingData.csv"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

SCHEMA_FILE_PATH:str = os.path.join("data_schema","schema.yaml")


DATA_INGESTION_COLLECTION_NAME:str =  "PhishingData"
DATA_INGESTION_DATABASE_NAME:str = "NetworkSecurity"
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION:float = 0.2



"""Starting with Data Validation Constants"""

DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALIDATED_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"


