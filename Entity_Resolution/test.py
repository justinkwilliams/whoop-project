import numpy as np
import pandas as pd
import res_functions as rf

df = pd.read_csv(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Entity_Resolution/fda_data.csv')
raw_names = df[['applicant']].copy()
test_df = raw_names.head(30).copy()

test_df
