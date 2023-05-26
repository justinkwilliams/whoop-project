import pandas as pd
import sqlalchemy
from res_functions import df_with_features, rows_to_pairs

# Create sqlalchemy engine
engine = sqlalchemy.create_engine(
    'sqlite:////Users/user/Desktop/Real Whoop Project/whoop_project/Data/fda_sqlalchemy.db')

# Query and create raw companies dataframe
query1 = "SELECT * FROM companies"
raw_comp = pd.read_sql(query1, engine)

# Create DataFrame for comparison
df = rows_to_pairs(raw_comp)

# Create DataFrame with features
df_feat = df_with_features(df, 'name1', 'name2')

# Create is match column based on 'jaro_score'
df_feat['ismatch'] = df_feat['jaro_score'].apply(
    lambda x: 1 if x > 0.9 else 0)

# Create DataFrame with matching values
matching_df = df_feat[df_feat['ismatch'] == 1].copy()

# Create DataFrame with matching id pars
id_pairs_df = matching_df[['id1', 'id2']]
