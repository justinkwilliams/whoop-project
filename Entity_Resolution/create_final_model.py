import pickle

import pandas as pd
from xgboost_classifier import predict_proba, train_model

#Read in DataFrame
df = pd.read_csv(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/training_set.csv')

#Set seed #
seed = 456

# Train model
xgb_model = train_model(
    df[['set_overlap', 'jaro_score', 'part_ratio']], df[['ismatch']], seed)

y_pred_proba = predict_proba(
    xgb_model, df[['set_overlap', 'jaro_score', 'part_ratio']])

# Save the trained model
filename = 'final_xgboost_class.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))
