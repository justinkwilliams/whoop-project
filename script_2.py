import functions_2 as f2
import xgboost_classifier as xg

df = f2.read_data(
    '/Users/user/Desktop/Real Whoop Project/whoop-project/Data/whoop_data.csv')


f2.clean_dataframe(df, df['Date'], ['Skin Temperature', 'Blood Oxygen'], [
                   'Strain Score', 'Recovery Score'], '2022-01-01', '2022-12-31')
