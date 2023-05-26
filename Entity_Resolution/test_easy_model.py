import pickle

import pandas as pd
import sqlalchemy
from res_functions import df_with_features
from xgboost_classifier import predict_proba

# Create sqlalchemy engine
engine = sqlalchemy.create_engine(
    'sqlite:////Users/user/Desktop/Real Whoop Project/whoop_project/Data/fda_sqlalchemy.db')

# Query and create raw companies dataframe
query1 = "SELECT * FROM companies"
raw_comp = pd.read_sql(query1, engine)

# Sample from table and create new DataFrame
comp_name1 = raw_comp.sample(n=50, replace=True, random_state=456)
comp_name1 = comp_name1.rename(
    columns={'id': 'id_1', 'company_name': 'name1'}).reset_index(drop=True)
comp_name2 = raw_comp.sample(n=50, replace=True, random_state=789)
comp_name2 = comp_name2.rename(
    columns={'id': 'id_2', 'company_name': 'name2'}).reset_index(drop=True)

# Create new DataFrame with features
merged_df = pd.concat([comp_name1, comp_name2], axis=1)

# Create DataFrame with features
df_feat = df_with_features(merged_df, 'name1', 'name2')

# Set seed value
seed = 123

# Load model
loaded_model = pickle.load(open(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Entity_Resolution/xgboost_classf.pkl', 'rb'))

# Predict Probabilities
y_pred_proba = predict_proba(
    loaded_model, df_feat[['set_overlap', 'jaro_score', 'part_ratio']])

# Add "proba" column to DataFrame
df_proba = df_feat.copy()
df_proba = df_proba.reset_index(drop=True)
just_proba = pd.DataFrame(y_pred_proba, columns=['proba'])
final_df = pd.concat([df_proba, just_proba], axis=1)

# Add manual 'ismatch' column
is_match_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
final_df['ismatch'] = is_match_list

# Save DataFrame to csv
final_df.to_csv(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/training_set.csv', index=False)

# df_with_proba = pd.DataFrame(X_test)
# df_with_proba['ismatch'] = y_test
# test_index = df_with_proba.index
# names_df = df.loc[test_index]
# df_with_proba = pd.concat([df_with_proba, names_df], axis=0)
# df_with_proba = df_with_proba.iloc[21:]
# proba_df_reset = df_with_proba.reset_index(drop=True)

# just_proba = pd.DataFrame(y_pred_proba, columns=['proba'])
# proba_df_reset = pd.concat([proba_df_reset, just_proba], axis=1)
# proba_df_reset
