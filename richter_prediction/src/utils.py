import os
import pandas as pd
import numpy as np


def count_obj_cols(df):
    obj_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            obj_cols.append(col)
    
    return obj_cols

def preprocess_x(df, encoder, scaler):
    obj_cols = count_obj_cols(df)
    df_int_cols = df.drop(obj_cols, axis=1, inplace=False)
    encoded_df = encoder.transform(df[obj_cols])
    encoded_df = pd.concat([df_int_cols, encoded_df], axis=1)
    scaled_df = scaler.transform(encoded_df)
    
    return scaled_df