from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkModel:
    def __init__(self,preprocessor_object,model_object):
        try:
            self.preprocessor = preprocessor_object
            self.model = model_object
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def predict(self,X):
        try:
            transformed_feature = self.preprocessor.transform(X)
            model_prediction = self.model.predict(transformed_feature)
            return model_prediction
        except Exception as e:
            raise NetworkSecurityException(e,sys)