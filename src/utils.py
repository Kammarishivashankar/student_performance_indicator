import sys
import os
import numpy as np
import pandas as pd
import dill

from src.exception import CustomerException
from src.logger import logging
from sklearn.metrics import r2_score



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomerException(e,sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models_dict):
    try:
        report = {}
        models_used = []
        test_score_list = []
        train_score_list = []

        for i in range(len(models_dict)):
            model = list(models_dict.values())[i]

            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            models_used.append(model)
            test_score_list.append(test_model_score)
            train_score_list.append(train_model_score)
        report['model'] = models_used
        report['train_model_score'] = train_score_list
        report['test_model_score'] = test_score_list
        return report
    except Exception as e:
        raise CustomerException(e,sys)

    



