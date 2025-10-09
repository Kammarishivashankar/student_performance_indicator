import sys
import pandas as pd
from src.exception import CustomException
import dill
from src.utils import load_object
import os
DEBUG = True

class PredictionDataConfig:
    prediction_data_path = os.path.join('Data',"predictionData")

class PredictPipeline:
    def __init__(self):
        pass
        
    def predict(self,features):
        try:
            model_path=os.path.join("models","model.pkl")
            preprocessor_path=os.path.join('models','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        self.prediciton_data_config = PredictionDataConfig()
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            prediction_data = pd.DataFrame(custom_data_input_dict)
            prediction_data.to_csv(self.prediciton_data_config.prediction_data_path)
            return prediction_data

        except Exception as e:
            raise CustomException(e, sys)
        
if DEBUG:
    if __name__ == "__main__":
        obj = CustomData('male','group D',"master's degree","standard","none",80,80)
        pred_data = obj.get_data_as_data_frame()
        
        pred_pipe_obj = PredictPipeline()
        pred = pred_pipe_obj.predict(pred_data)
        print(pred)



        
