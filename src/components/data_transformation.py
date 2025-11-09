import sys
import os
from src.exception import customException
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object
from src.logger import logging

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class dataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_cols = ["writing_score","reading_score"]
            categorical_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")), #for handling the missing values
                    ("scaler",StandardScaler()) #for standard scaling - 
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_hot_encode",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]


            )
            logging.info("Numerical Cols Standard Scaling completed")

            logging.info("Categorical Cols Encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_cols),     
                    ("cat_pipeline",categorical_pipeline,categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise customException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("The train and test data read successfully.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_cols = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and test dataframe"
            )

            input_feture_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feture_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr =np.c_[input_feture_train_arr,np.array(target_feature_train_df)]
            
            test_arr = np.c_[input_feture_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise customException(e,sys)
               

