import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

debug=True

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('Data',"Raw_data.csv")
    train_data_path: str=os.path.join('Data','train.csv')
    test_data_path: str=os.path.join("Data",'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self):

        logging.info('DataIngestion.start_data_ingestion started...!')
        try:
            df = pd.read_csv(os.path.join('EDA','StudentsPerformance.csv'))
            df.columns = [col.replace(" ",'_').replace('/',"_") for col in df.columns]
            logging.info('Read the dataset in DataIngestion.start_data_ingestion..')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('raw data saved to csv in DataIngestion.start_data_ingestion..')

            logging.info('Train test split started in DataIngestion.start_data_ingestion....')
            train_set, test_set = train_test_split(df,test_size=0.25,random_state=0)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('training and testing data saved to csv in DataIngestion.start_data_ingestion..')

            logging.info('Ingestion of data is completed!')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if debug:
    if __name__ == "__main__":
        obj = DataIngestion()
        train_data,test_data  = obj.start_data_ingestion()

        data_trf_obj = DataTransformation()
        x_train,y_train,x_test,y_test = data_trf_obj.start_data_transformation(train_data,test_data)    
        print(x_train.head())
        
        # model_trainer_obj = ModelTrainer()
        # score = model_trainer_obj.start_model_trainer(x_train,y_train,x_test,y_test)
        # print(score)



 

