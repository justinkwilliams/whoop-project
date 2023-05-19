import pickle

import pandas as pd
from res_functions import df_with_features
from xgboost_classifier import (assess_part, predict, predict_proba,
                                split_data, train_model)

# Create df with easy data
name1 = ['Apple', 'Grape', 'turtle', 'Steelers', 'phone',
         'plug', 'badge', 'sleep', 'lunch', 'door handle', 'credit card', 'water', 'speakers', 'line', 'pad', 'shelf', 'pen', 'screen', 'camping', 'enter', 'tube', 'upmc']

name2 = ['Apple', 'Grape', 'fish', 'steelers', 'computer',
         'charger', 'boat', 'sleep', 'dinner', 'door handle', 'carrot cake', 'bottle', 'speakers', 'line', 'pad', 'stage', 'light', 'screen', 'camera', 'return', 'tube', 'upmc']

ismatch = [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]

data = {'name1': name1, 'name2': name2, "ismatch": ismatch}

easy_df = pd.DataFrame(data)

# Create DataFrame with features
df = df_with_features(easy_df, 'name1', 'name2')

# Create constant seed/random_state
seed = 148

# Split data
X_train, X_test, y_train, y_test = split_data(
    df, ['set_overlap', 'jaro_score', 'part_ratio'], 'ismatch', 0.7, seed)

# Train model
xgb_model = train_model(X_train, y_train, seed)

# Save the trained model
filename = 'xgboost_class.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))

# Predict probabilities
y_pred_proba = predict_proba(xgb_model, X_test)

y_pred_proba
