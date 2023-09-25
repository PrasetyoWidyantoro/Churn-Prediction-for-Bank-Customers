#import all realated libraries
#import libraries for data analysis
import numpy as np
import pandas as pd

# import library for visualization
import matplotlib.pyplot as plt

# import pickle and json file for columns and model file
import pickle
import json
import joblib
import yaml
import scipy.stats as scs

# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")

# library for model selection and models
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

# evaluation metrics for classification model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime
from sklearn.metrics import classification_report
import uuid

from tqdm import tqdm
import pandas as pd
import os
import copy
import yaml
import joblib
import util as util
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

###################################################

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_data_scaling(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    #Read data X_train dan y_sm hasil dari data preparation
    X_ros_clean = util.pickle_load(config_data["standar_scaler_ros"][0])
    y_ros = util.pickle_load(config_data["standar_scaler_ros"][1])

    #Read data X_valid dan y_valid hasil dari data preparation
    X_valid_clean = util.pickle_load(config_data["standar_scaler_valid"][0])
    y_valid = util.pickle_load(config_data["standar_scaler_valid"][1])

    #Read data X_test dan y_test hasil dari data preparation
    X_test_clean = util.pickle_load(config_data["standar_scaler_test"][0])
    y_test = util.pickle_load(config_data["standar_scaler_test"][1])

    # Return 3 set of data
    return X_ros_clean, y_ros, X_valid_clean, y_valid, X_test_clean, y_test


def binary_classification_lgbm_tuned(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # set hyperparameters for tuning
    # instantiate the classifier
    best_lgbm_clf = lgb.LGBMClassifier(random_state=123)
    
    best_lgbm_clf.fit(X_ros_clean, y_ros)
    
    # evaluate on validation set
    valid_pred = best_lgbm_clf.predict(x_valid)
    report = classification_report(y_valid, valid_pred, output_dict=True)
    valid_recall = report['weighted avg']['recall']
    print('Validation recall:', valid_recall)
    
    # evaluate on test set
    test_pred = best_lgbm_clf.predict(x_test)
    report = classification_report(y_test, test_pred, output_dict=True)
    test_recall = report['weighted avg']['recall']
    print('Test recall:', test_recall)
    
    return best_lgbm_clf

def save_model_log(model, model_name, X_test, y_test):
    # generate unique id
    model_uid = uuid.uuid4().hex
    
    # get current time and date
    now = time_stamp()
    training_time = now.strftime("%H:%M:%S")
    training_date = now.strftime("%Y-%m-%d")
    
    # generate classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # create dictionary for log
    log = {"model_name": model_name,
           "model_uid": model_uid,
           "training_time": training_time,
           "training_date": training_date,
           "classification_report": report}
    
    # menyimpan log sebagai file JSON
    with open('training_log/training_log.json', 'w') as f:
        json.dump(log, f)
        
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()
    
    # 2. Load dataset
    X_ros_clean, y_ros, X_valid_clean, y_valid, X_test_clean, y_test = load_data_scaling(config_data)
    
    lgbm_best = binary_classification_lgbm_tuned(x_train=X_ros_clean, y_train=y_ros,
                                                 x_valid=X_valid_clean, y_valid=y_valid,
                                                 x_test=X_test_clean, y_test=y_test)
    
    save_model_log(model = lgbm_best, model_name = "LightGBM Default", X_test = X_test_clean, y_test=y_test)
    
    lightGBM_default = config_data["model_final"]
    with open(lightGBM_default, 'wb') as file:
        pickle.dump(lgbm_best, file)