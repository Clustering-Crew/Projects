import os
import joblib
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from category_encoders import CountEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import catboost
import xgboost

from utils import *

def count_obj_cols(df):
    obj_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            obj_cols.append(col)
    
    return obj_cols

def preprocess_x(df):
    obj_cols = count_obj_cols(df)
    df_int_cols = df.drop(obj_cols, axis=1, inplace=False)
    encoded_df = encoder.transform(df[obj_cols])
    encoded_df = pd.concat([df_int_cols, encoded_df], axis=1)
    scaled_df = scaler.transform(encoded_df)
    
    return scaled_df

def train(X, Y, model, number):

    building_id = X["building_id"]
    X.drop("building_id", axis=1, inplace=True)
    Y.drop("building_id", axis=1, inplace=True)
    
    scaled_x = preprocess_x(X)
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, Y, test_size=0.2)
    
    model.fit(x_train, y_train)
    joblib.dump(model, f"../../proj_datasets/richter_prediction/assets/model_{number}.pkl")
    y_pred = model.predict(x_test)
    print(f"F1 score before feature selection - {f1_score(y_test, y_pred, average='micro')}")



def predict(path, model_path, number):
    df = pd.read_csv(path)
    building_id = np.array(df['building_id'].array)
    df.drop('building_id', axis=1, inplace=True)
    model = joblib.load(model_path)
    scaled_df = preprocess_x(df)
    prediction = model.predict(scaled_df)
    prediction = np.array(prediction)
    prediction = np.reshape(prediction, (prediction.shape[0],))

    result_df = pd.DataFrame({'building_id': building_id, "damage_grade": prediction})

    result_df.to_csv(f"../../proj_datasets/richter_prediction/submission/submission_{number}.csv", index=False)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int)
    parser.add_argument("--predict", type=bool)
    parser.add_argument("--test_data", type=str, default="test.csv")
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()

    # Load model, data, encoder and scaler
    DATASET_PATH = "../../proj_datasets/richter_prediction"
    ASSETS_PATH = "../../proj_datasets/richter_prediction/assets"

    X = pd.read_csv(os.path.join(DATASET_PATH, "train_values.csv"))
    Y = pd.read_csv(os.path.join(DATASET_PATH, "train_labels.csv"))

    encoder = joblib.load(os.path.join(ASSETS_PATH, "count_encoder.pkl"))
    scaler = joblib.load(os.path.join(ASSETS_PATH, "standardscaler.pkl"))

    model = xgboost.XGBClassifier()
    model_path = f"../../proj_datasets/richter_prediction/assets/model_{args.num}.pkl"

    if args.predict:
        predict(args.test_data, model_path, args.num)
    else:
        train(X, Y, model, args.num)            