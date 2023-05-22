import pickle

import numpy as np
import pandas as pd
import sqlalchemy
import xgboost
from res_functions import df_with_features
from xgboost_classifier import (assess_part, predict, predict_proba,
                                split_data, train_model)

# Create sqlalchemy engine
engine = sqlalchemy.create_engine(
    'sqlite:////Users/user/Desktop/Real Whoop Project/whoop_project/Data/fda_sqlalchemy.db')

# Query and create raw companies dataframe
query1 = "SELECT * FROM companies"
raw_comp = pd.read_sql(query1, engine)

# Sample from table and create new DataFrame
comp_name1 = raw_comp.sample(n=10000, replace=True, random_state=123)
comp_name1 = comp_name1.rename(
    columns={'id': 'id_1', 'company_name': 'name1'}).reset_index(drop=True)
comp_name2 = raw_comp.sample(n=10000, replace=True, random_state=456)
comp_name2 = comp_name2.rename(
    columns={'id': 'id_2', 'company_name': 'name2'}).reset_index(drop=True)

# Create new DataFrame with features
merged_df = pd.concat([comp_name1, comp_name2], axis=1)

# Create DataFrame with features
df_feat = df_with_features(merged_df, 'name1', 'name2')

# Set seed value
seed = 123

# Split data
X_train, X_test, y_train, y_test = split_data(
    df_feat, ['set_overlap', 'jaro_score', 'part_ratio'], 'name1', 0.7, seed)

# Load model
loaded_model = pickle.load(open(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Entity_Resolution/xgboost_classf.pkl', 'rb'))

# Predict Probabilities

y_pred_proba = predict_proba(loaded_model, X_test)
