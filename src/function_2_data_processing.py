from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import copy
import joblib
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import datetime
import util as util


config_dir = "config/config.yaml"

############################################
def load_data(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    X_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    X_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Return 3 set of data
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    X_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    X_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
        [X_train, y_train],
        axis = 1
    )
    valid_set = pd.concat(
        [X_valid, y_valid],
        axis = 1
    )
    test_set = pd.concat(
        [X_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set


def imputeData(data, numerical_columns_mean, numerical_columns_median, categorical_columns):
    """
    Fungsi untuk melakukan imputasi data numerik dan kategorikal
    :param data: <pandas dataframe> sample data input
    :param numerical_columns_mean: <list> list kolom numerik data yang akan diimputasi dengan mean
    :param numerical_columns_median: <list> list kolom numerik data yang akan diimputasi dengan median
    :param categorical_columns: <list> list kolom kategorikal data
    :return numerical_data_imputed: <pandas dataframe> data numerik imputed
    :return categorical_data_imputed: <pandas dataframe> data kategorikal imputed
    """
    # Imputasi kolom numerik dengan mean
    numerical_data_mean = data[numerical_columns_mean]
    imputer_numerical_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_numerical_mean.fit(numerical_data_mean)
    imputed_data_mean = imputer_numerical_mean.transform(numerical_data_mean)
    numerical_data_imputed_mean = pd.DataFrame(imputed_data_mean, columns=numerical_columns_mean, index=numerical_data_mean.index)

    # Imputasi kolom numerik dengan median
    numerical_data_median = data[numerical_columns_median]
    imputer_numerical_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer_numerical_median.fit(numerical_data_median)
    imputed_data_median = imputer_numerical_median.transform(numerical_data_median)
    numerical_data_imputed_median = pd.DataFrame(imputed_data_median, columns=numerical_columns_median, index=numerical_data_median.index)

    # Gabungkan kedua data numerik yang telah diimputasi
    numerical_data_imputed = pd.concat([numerical_data_imputed_mean, numerical_data_imputed_median], axis=1)

    # Seleksi data kategorikal
    categorical_data = data[categorical_columns]

    # Imputasi dengan menggunakan modus
    mode = categorical_data.mode().iloc[0]

    # Lakukan imputasi untuk data kategorikal
    categorical_data_imputed = categorical_data.fillna(mode)

    # Gabungkan data numerik dan kategorikal yang telah diimputasi
    data_imputed = pd.concat([numerical_data_imputed, categorical_data_imputed], axis=1)

    return data_imputed

def get_dummies(train_df, input_df):
    # Menggabungkan data train dan input menjadi satu DataFrame
    combined_df = pd.concat([train_df, input_df])
    
    # Mengubah variabel kategorikal menjadi variabel dummy
    dummies_df = pd.get_dummies(combined_df, columns=train_df.select_dtypes(include='object').columns)
    
    # Memisahkan kembali data train dan input
    train_dummies = dummies_df[:train_df.shape[0]]
    input_dummies = dummies_df[train_df.shape[0]:]
    
    return train_dummies, input_dummies

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 123)

    # Balancing set data
    x_ros, y_ros = ros.fit_resample(
        set_data.drop("Exited", axis = 1),
        set_data.Exited
    )

    # Concatenate balanced data
    set_data_ros = pd.concat(
        [x_ros, y_ros],
        axis = 1
    )

    # Return balanced data
    return set_data_ros


columns_to_scale = ["CreditScore", "Age", "EstimatedSalary", "Balance"]
#"Tenure",
#Fungsi untuk fitting scaling
def fit_scaler(train_data):
    config_data = util.load_config()
    scaler = StandardScaler()
    scaler.fit(train_data.loc[:, columns_to_scale])
    # save scaler
    #with open('model/5 - Model Final/scaler.pkl', 'wb') as f:
    with open(config_data["model_scaler"], 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

#Fungsi untuk melakukan pemanggilan untuk model scaler.pkl yang telah dibuat
def load_scaler(folder_path):
    # load scaler
    file_path = os.path.join(folder_path, 'scaler.pkl')
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# fungsi transform data untuk scaling data menggunakan standartscaler
def transform_data(data, scaler):
    scaled_data = scaler.transform(data.loc[:, columns_to_scale])
    data.loc[:, columns_to_scale] = scaled_data
    return data


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(config_data)
    
    #Imputasi Data
    numerical_column = ["Age"]
    numerical_column_mean = ["CreditScore", "Balance", "EstimatedSalary"]
    set_numerik = numerical_column + numerical_column_mean
    dataset_column = list(X_train.columns)
    categorical_column = list(set(dataset_column).difference(set(set_numerik)))
    
    X_train_impute = imputeData(data = X_train, 
                                numerical_columns_mean = numerical_column_mean, 
                                numerical_columns_median = numerical_column, 
                                categorical_columns = categorical_column)
    
    X_valid_impute = imputeData(data = X_valid, 
                                numerical_columns_mean = numerical_column_mean, 
                                numerical_columns_median = numerical_column, 
                                categorical_columns = categorical_column)
    
    X_test_impute = imputeData(data = X_test, 
                               numerical_columns_mean = numerical_column_mean, 
                               numerical_columns_median = numerical_column, 
                               categorical_columns = categorical_column)
    
    util.pickle_dump(X_train_impute, config_data["impute_data_train"][0])
    util.pickle_dump(y_train, config_data["impute_data_train"][1])
    util.pickle_dump(X_test_impute, config_data["impute_data_test"][0])
    util.pickle_dump(y_test, config_data["impute_data_test"][1])
    util.pickle_dump(X_valid_impute, config_data["impute_data_valid"][0])
    util.pickle_dump(y_valid, config_data["impute_data_valid"][1])
    
    #ohe
    dataset_ohe, valid_set = get_dummies(X_train_impute, X_valid_impute)
    dataset, test_set = get_dummies(X_train_impute, X_test_impute)

    # 12. SMOTE dataset
    dataset = pd.concat([dataset, y_train],axis = 1)
    dataset_ros = ros_fit_resample(dataset)
    #Fitting Scaler
    scaler = fit_scaler(dataset_ros)
    scaling = load_scaler(config_data["scaler"])
    # scaling = load_scaler('model/5 - Model Final/')
    # transform selected columns of training data
    
    dataset_clean = transform_data(dataset_ros, scaling)
    X_valid_clean = transform_data(valid_set, scaling)
    X_test_clean = transform_data(test_set, scaling)
    
    X_ros_clean = dataset_clean.drop(columns = "Exited")
    y_ros = dataset_clean["Exited"]
    
    X_ros_clean = X_ros_clean[sorted(X_ros_clean.columns)]
    X_valid_clean = X_valid_clean[sorted(X_valid_clean.columns)]
    X_test_clean = X_test_clean[sorted(X_test_clean.columns)]

    util.pickle_dump(X_ros_clean, config_data["standar_scaler_ros"][0])
    util.pickle_dump(y_ros, config_data["standar_scaler_ros"][1])

    util.pickle_dump(X_test_clean, config_data["standar_scaler_test"][0])
    util.pickle_dump(y_test, config_data["standar_scaler_test"][1])

    util.pickle_dump(X_valid_clean, config_data["standar_scaler_valid"][0])
    util.pickle_dump(y_valid, config_data["standar_scaler_valid"][1])