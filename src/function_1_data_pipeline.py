from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import joblib
import yaml
from datetime import datetime
import util as util
import numpy as np
# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
# import pickle and json file for columns and model file
import pickle
import json
import copy

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input_data, params):
    # Check data types
    assert input_data.select_dtypes("float").columns.to_list() == params["float64_columns"], "an error occurs in datetime column(s)."
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."

    # Check range of data
    assert set(input_data.Geography).issubset(set(params["range_Geography"])), "an error occurs in stasiun range."
    assert set(input_data.Gender).issubset(set(params["range_Gender"])), "an error occurs in stasiun range."
    assert input_data.CreditScore.between(params["range_CreditScore"][0], params["range_CreditScore"][1]).sum() == len(input_data), "an error occurs in pm10 range."
    assert input_data.Age.between(params["range_Age"][0], params["range_Age"][1]).sum() == len(input_data), "an error occurs in pm25 range."
    assert input_data.Tenure.between(params["range_Tenure"][0], params["range_Tenure"][1]).sum() == len(input_data), "an error occurs in so2 range."
    assert input_data.Balance.between(params["range_Balance"][0], params["range_Balance"][1]).sum() == len(input_data), "an error occurs in co range."
    assert input_data.NumOfProducts.between(params["range_NumOfProducts"][0], params["range_NumOfProducts"][1]).sum() == len(input_data), "an error occurs in o3 range."
    assert input_data.HasCrCard.between(params["range_HasCrCard"][0], params["range_HasCrCard"][1]).sum() == len(input_data), "an error occurs in no2 range."
    assert input_data.IsActiveMember.between(params["range_IsActiveMember"][0], params["range_IsActiveMember"][1]).sum() == len(input_data), "an error occurs in no2 range."
    assert input_data.EstimatedSalary.between(params["range_EstimatedSalary"][0], params["range_EstimatedSalary"][1]).sum() == len(input_data), "an error occurs in o3 range."
    #assert input_data.Exited.between(params["Exited_categories"][0], params["Exited_categories"][1]).sum() == len(input_data), "an error occurs in no2 range."

######################################################################################################################    

if __name__ == "__main__":
    # Load configuration file
    config_data = util.load_config()
    
    # Read all raw Dataset
    raw_dataset = read_raw_data(config_data).drop(["RowNumber","CustomerId","Surname"], axis = 1)
    
    #3.Check Dataset
    check_data(raw_dataset, config_data)
    
    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # Splitting input output
    X = raw_dataset.drop(columns="Exited")
    y = raw_dataset["Exited"]

    #6. Splitting train test
    #Split Data 70% training 30% testing
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.3, 
        random_state = 123)
    
    #6. Splitting test valid
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size = 0.4,
        random_state = 42,
        stratify = y_test
    )
    
    #Menggabungkan x train dan y train untuk keperluan EDA
    util.pickle_dump(X_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(X_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(X_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
    
    print("Data Pipeline passed successfully.")
