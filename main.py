from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig

import sys



if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        DataIngestion = DataIngestion(dataingestionconfig)
        logging.info("Starting the data ingestion process")
        dataingestionartifact = DataIngestion.initiate_data_ingestion()

        print(dataingestionartifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)