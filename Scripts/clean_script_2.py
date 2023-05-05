import whoop_project.clean_functions as cf

# Read in csv file
df = cf.read_data(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/whoop_data.csv')
# Clean DataFrame
df = cf.clean_dataframe(df, 'Date')

# Convert "Nap Count" column to bool
cf.nap_bool(df, 'nap_count')

# Create new recovery category column
cf.recovery_cat(df, 'recovery_score')


# Check DataFrame shape, columns
df.shape
df.head()
df.columns
df.dtypes

# Plot missing data
cf.plot_missing_data(df)

# Fill missing values?

# Save dataframe as clean dataframe
df.to_csv('whoop_extra_col.csv', index=False)
