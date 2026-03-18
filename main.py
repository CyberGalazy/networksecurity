from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataValidationConfig
import sys



if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        DataIngestion = DataIngestion(dataingestionconfig)
        logging.info("Starting the data ingestion process")
        dataingestionartifact = DataIngestion.initiate_data_ingestion()
        logging.info("Data ingestion process completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Starting the data validation process")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation process completed")
        print(data_validation_artifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)