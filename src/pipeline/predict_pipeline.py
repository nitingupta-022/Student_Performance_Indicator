import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline : 
    '''
    This class is responsible for loading the trained model and preprocessor, and making predictions on new data.
    '''
    
    def __init__(self) : 
        pass

    def predict(self, features) : 
        try : 
            logging.info("Prediction request received")
            preprocessor_path = "artifacts/preprocessor.pkl"
            hyperparameter_path = "artifacts/tuned_model.pkl"

            preprocessor = load_object(file_path = preprocessor_path)
            model = load_object(file_path = hyperparameter_path)
            logging.info("Artifacts loaded")

            data_scaled = preprocessor.transform(features)
            logging.info("Input transformed")
            preds = model.predict(data_scaled)
            logging.info("Prediction finished")

            return preds

        except Exception as e :
            raise CustomException(e, sys)


class CustomData : 
    '''
    This class is responsible for taking input data from the user and converting it into a format that can be used for prediction.
    '''
    
    def __init__(self, 
            gender: str, 
            race_ethnicity: str, 
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int ) : 

        self.gender = gender        
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self) : 
        try : 
            logging.info("Preparing custom input data for prediction")
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom input dataframe created successfully with columns=%s", list(df.columns))
            return df
        
        except Exception as e :
            raise CustomException(e, sys)