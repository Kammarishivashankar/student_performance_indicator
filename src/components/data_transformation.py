import numpy as np
import pandas as pd
import sys
import os

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomerException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('models',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data__transformation_config=DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ['writing score',"reading score"]
            categorical_columns = ['gender',
                                   'race/ethnicity',
                                   'parental level of education',
                                   'lunch',
                                   'test preparation course']
            
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequenct')),
                    ("One_hot_encoder",OneHotEncoder())
                    ("scaler",StandardScaler())
                ]
            )


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_pipeline",categorical_pipeline,categorical_columns)
                ],
                remainder='passthrough'
            )

            logging.info('Data transformation completed!....')
            return preprocessor
        
        except Exception as e:
            logging.info('Data transformation failed!....')
            raise CustomerException(e,sys)
        
    def start_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Data read for training and testing is done!')

            prepprocessor_obj = self.get_data_transformer()

            target = "math score"
            

        except Exception as e:
            logging.info(' failed!....')
            raise CustomerException(e,sys)

        


