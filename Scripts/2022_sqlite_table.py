import pandas as pd
import sqlite3
# Read in CSV to DataFrame
df = pd.read_csv(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/2022_clean_df.csv')

# Rename some columns
df = df.rename(columns={'date': 'date_col',
                        'sleep_disturbances_/_hour': 'sleep_disturbances_per_hour'})

# Connect to database
conn = sqlite3.connect('whoop_project.db')

# Create dictionary with correct dtypes
dtypes = {
    'date_col': 'DATE',
    'strain_score': 'REAL',
    'skin_temperature': 'REAL',
    'avg_heart_rate': 'INTERGER',
    'heart_rate_variability': 'REAL',
    'max_heart_rate': 'INTEGER',
    'resting_heart_rate': 'INTEGER',
    'blood_oxygenation': 'REAL',
    'total_calories_burned': 'INTEGER',
    'recovery_score': 'INTEGER',
    'deep_sleep': 'REAL',
    'deep_sleep_percent': 'REAL',
    'light_sleep': 'REAL',
    'light_sleep_percent': 'REAL',
    'nap_count': 'INTEGER',
    'nap_sleep': 'REAL',
    'rem_sleep': 'REAL',
    'rem_sleep_percent': 'REAL',
    'sleep_end_time': 'REAL',
    'sleep_score': 'INTEGER',
    'sleep_start_time': 'REAL',
    'time_awake': 'REAL',
    'time_in_bed': 'REAL',
    'total_sleep': 'REAL',
    'sleep_consistency': 'INTEGER',
    'sleep_cycles': 'INTEGER',
    'sleep_disturbances': 'INTEGER',
    'sleep_disturbances_per_hour': 'REAL',
    'sleep_disturbances_duration': 'REAL',
    'sleep_onset_latency': 'REAL',
    'sleep_respiratory_rate': 'REAL',
    'time_asleep_percent': 'REAL',
    'time_awake_percent': 'REAL',
    'day_of_week': 'VARCHAR(10)',
    'nap': 'BOOLEAN',
    'recovery_level_cat': 'VARCHAR(10)'
}
# Create Table with all columns and rows
df.to_sql('raw_2022', conn,
          if_exists='replace', index=False, dtype=dtypes)


# Commit and close connection
conn.commit()
conn.close()



