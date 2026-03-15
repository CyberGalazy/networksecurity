import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.logging import logger
from networksecurity.exception import exception 


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise exception.NetworkSecurityException(e,sys)
    def csv_to_json(self,file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True,inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise exception.NetworkSecurityException(e,sys)
    
    def push_data_to_mongodb(self,records,database,collection_name):
        try:
            self.database = database
            self.collection_name = collection_name
            self.records = records
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.db = self.mongo_client[self.database]
            self.collection = self.db[self.collection_name]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise exception.NetworkSecurityException(e,sys)
        
if __name__=='__main__':
    FILE_PATH = "Network_Data\phisingData.csv"
    DATABASE = "NetworkSecurity"
    Collection = "PhishingData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json(FILE_PATH)
    print(f"Number of records extracted from csv file: {len(records)}")
    no_of_records = networkobj.push_data_to_mongodb(records,DATABASE,Collection)
    print(f"Number of records pushed to MongoDB: {no_of_records}")
