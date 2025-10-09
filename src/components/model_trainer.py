import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join('models',"model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def start_model_trainer(self,x_train,y_train,x_test,y_test):
        try:            
            models_dict = {
                "DecisionTree" : DecisionTreeRegressor(),
                "RandomForest" : RandomForestRegressor(),
                "GradientBoost" : GradientBoostingRegressor(),
                "XGBoost" : XGBRegressor(),
                "CatBoost" : CatBoostRegressor(verbose=False),
                "AdaBoost" : AdaBoostRegressor(),
                "KNeighours" : KNeighborsRegressor(),
                "LinearRegreesion" : LinearRegression(),
                }
            
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighours":{},
                "LinearRegreesion":{},
                
            }           
            
            model_report :dict = evaluate_models(x_train,y_train,x_test,y_test,models_dict,params)

            model_report_df = pd.DataFrame(model_report)
            best_model_row = model_report_df.sort_values(by='test_model_score', ascending=False).iloc[0]
            best_model = best_model_row['model']
            best_score = best_model_row['test_model_score']
            if best_score < 0.6:
                raise CustomException('No best model found!!')
            logging.info(f'Best model found on both {best_model} with score {best_score}')
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            return best_score
        except Exception as e:
            raise CustomException(e,sys)
        

        