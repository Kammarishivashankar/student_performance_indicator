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

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('models',"preprocessor.pkl")
    processed_train_data :str = os.path.join("Data",'Processed_train_data')
    processed_test_data :str = os.path.join("Data",'Processed_test_data')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ['writing_score',"reading_score"]
            categorical_columns = ['gender',
                                'race_ethnicity',
                                'parental_level_of_education',
                                'lunch',
                                'test_preparation_course']
            
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
                remainder='passthrough',
                verbose_feature_names_out=False
            )

            logging.info('Data transformation completed!....')
            return preprocessor
        
        except Exception as e:
            logging.info('Data transformation failed!....')
            raise CustomException(e,sys)
        
    def start_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Data read for training and testing is done!')

            prepprocessor_obj = self.get_data_transformer()
            prepprocessor_obj.set_output(transform='pandas')

            target = "math_score"

        
            input_train_data = train_df.drop(columns=[target])
            target_train_data = train_df[target]

            input_test_data = test_df.drop(columns=[target])
            target_test_data = test_df[target]

            logging.info(f"training and testing test preparation done...")

            input_train_data_transfomed = prepprocessor_obj.fit_transform(input_train_data)
            input_test_data_transformed = prepprocessor_obj.transform(input_test_data)

            input_train_data_transfomed.to_csv(self.data_transformation_config.processed_train_data,index=False,header=True)
            input_test_data_transformed.to_csv(self.data_transformation_config.processed_test_data,index=False,header=True)

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=prepprocessor_obj
            )
            logging.info(f"saving preprocessor completed...")

            return (
                input_train_data_transfomed,
                target_train_data,
                input_test_data_transformed,
                target_test_data
            )
        except Exception as e:
            logging.info('start_data_transformation failed!....')
            raise CustomException(e,sys)