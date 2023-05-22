import pickle

import pandas as pd
from res_functions import df_with_features
from xgboost_classifier import predict_proba, split_data, train_model

# Create easy DataFrame
name1 = ['Apple', 'Grape', 'turtle', 'Steelers', 'phone',
         'plug', 'badge', 'sleep', 'lunch', 'door',
         'elephant', 'wheel', 'butterfly', 'lava', 'guitar',
         'shoe', 'banana', 'cloud', 'keychain', 'mirror',
         'hydro flask', 'mouse pad', 'ink pen', 'key chain', 'computer screen',
         'wrist watch', 'call button', 'blue mask', 'lunch box', 'school binder',
         'toe nail', 'ring finger', 'code type', 'caps lock', 'white truck',
         'book bag', 'lost bet', 'right exam', 'dollar bill', 'spider man',
         'good apple pie', 'large cheese pizza', 'small black notebook', 'mac air pods', 'very fast car',
         'sun cloud rain', 'boost model fun', 'rock hard desk', 'hand on watch', 'main focus spot',
         'water great toast', 'panic fling budge', 'leave bread brave', 'pawn link say', 'low spy fit',
         'allow south arise', 'write brain clock', 'smile dance ample', 'blame gap ride', 'level chain first',
         'Acme Corporation', 'Globex Incoporated', 'Stark Industries', 'Wayne Enterprises', 'Umbrella Corporation',
         'Cyberdyne Systems', 'InGen Corporation', 'Weyland-Yutani Corporation', 'Oscorp Industries', 'Aperture Science',
         ]

name2 = ['Apple', 'Grape', 'turtle', 'Steelers', 'phone',
         'plug', 'badge', 'sleep', 'lunch', 'door',
         'lion', 'tire', 'balloon', 'lamp', 'bass',
         'straw', 'band', 'clown', 'lock', 'glass',
         'hydro flask', 'mouse pad', 'ink pen', 'key chain', 'computer screen',
         'wrist watch', 'call button', 'blue mask', 'lunch box', 'school binder',
         'tote note', 'bed head', 'card top', 'cams log', 'black truck',
         'back pack', 'win bet', 'left exam', 'coin star', 'cat woman',
         'good apple pie', 'large cheese pizza', 'small black notebook', 'mac air pods', 'very fast car',
         'Sun Cloud Rain', 'boost Model fun', 'rock hard Desk', 'Hand On Watch', 'main Focus Spot',
         'craft croud fourm', 'rise hron shed', 'young float image', 'nut tread player', 'drop feed dark',
         'wallow sort argue', 'wright brow clerk', 'sample dance angel', 'brake rap role', 'metal cabin fruit',
         'Acme Corp', 'Globex Inc', 'Stark Industries', 'Wayne Ent', 'Umbrella Corp',
         'Cyberdyne Systems', 'inGen corporation', 'Weyland-Yutani', 'Oscorp Industries', 'Aperture Science corp'
         ]

ismatch = [1, 1, 1, 1, 1,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 1, 1, 1, 1]

data = {'name1': name1, 'name2': name2, "ismatch": ismatch}

easy_df = pd.DataFrame(data)


# Create DataFrame with features
df = df_with_features(easy_df, 'name1', 'name2')

# Create constant seed/random_state
seed = 123

# Split data
X_train, X_test, y_train, y_test = split_data(
    df, ['set_overlap', 'jaro_score', 'part_ratio'], 'ismatch', 0.7, seed)

# Train model
xgb_model = train_model(X_train, y_train, seed)

# Save the trained model
filename = 'xgboost_classf.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))

# Predict probabilities
y_pred_proba = predict_proba(xgb_model, X_test)

y_pred_proba
