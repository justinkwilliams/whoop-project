import whoop_project.clean_functions as cf

# Read in csv file
df = cf.read_data(
    '/Users/user/Desktop/Real Whoop Project/whoop-project/Data/whoop_data.csv')
# Clean DataFrame
df = cf.clean_dataframe(df, 'Date', ['Skin Temperature', 'Blood Oxygenation', 'Nap Sleep'], [
    'Strain Score', 'Recovery Score'], '1/1/22', '12/31/22')

# Convert "Nap Count" column to bool
cf.nap_bool(df, 'nap_count')

# Create new recovery category column
cf.recovery_cat(df, 'recovery_score')


# Check DataFrame shape, columns
df.shape
df.head()
df.columns
df.dtypes

# CPlot missing data
cf.plot_missing_data(df)

# Save dataframe as clean dataframe
df.to_csv('2022_clean_df.csv', index=False)
