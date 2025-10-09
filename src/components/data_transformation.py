import numpy as np
import pandas as pd
import sys
import os

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('models',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

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
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False))
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
            prepprocessor_obj.set_output(transform='pandas')

            target = "math score"

        
            input_train_data = train_df.drop(columns=[target])
            target_train_data = train_df[target]

            input_test_data = test_df.drop(columns=[target])
            target_test_data = test_df[target]

            logging.info(f"training and testing test preparation done...")

            input_train_data_transfomed = prepprocessor_obj.fit_transform(input_train_data)
            input_test_data_transformed = prepprocessor_obj.transform(input_test_data)

            

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=prepprocessor_obj
            )

            return (
                input_train_data_transfomed,
                input_test_data_transformed,
                target_train_data,
                target_test_data
            )
        except Exception as e:
            logging.info(' failed!....')
            raise CustomerException(e,sys)