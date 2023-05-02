import xgboost_regressor as xg
from clean_functions import read_data

# Load in clean data
df = read_data(
    '/Users/user/Desktop/Real Whoop Project/whoop-project/Data/2022_clean_df.csv')

# Seperate to testing and training sets
X_train, X_test, y_train, y_test = xg.split_data(df, ['strain_score', 'avg_heart_rate', 'heart_rate_variability',
                                                      'max_heart_rate', 'resting_heart_rate', 'total_calories_burned', 'deep_sleep', 'deep_sleep_percent', 'light_sleep',
                                                      'light_sleep_percent', 'nap_count', 'rem_sleep', 'rem_sleep_percent',
                                                      'sleep_end_time', 'sleep_score', 'sleep_start_time', 'time_awake',
                                                      'time_in_bed', 'total_sleep', 'sleep_consistency', 'sleep_cycles',
                                                      'sleep_disturbances', 'sleep_disturbances_/_hour',
                                                      'sleep_disturbances_duration', 'sleep_onset_latency',
                                                      'sleep_respiratory_rate', 'time_asleep_percent', 'time_awake_percent'], 'recovery_score', 0.7)

# Train model
xgb_model = xg.train_model(X_train, y_train)

# Model predictions
xg.predict(xgb_model, X_test)

# Assess model
assesments = xg.assess(xgb_model, X_test, y_test)
print(assesments)
