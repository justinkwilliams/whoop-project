import sqlite3

import pandas as pd


def create_rel_tables(path_to_db: str, queries: list[str]):
    """Sends queries to a desired database.

    Args:
        path_to_db (str): Path to database of interest.
        queries (list[str]): List of queries to pass to the database.
    """
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    for query in queries:
        cursor.execute(query)
    cursor.close()


# Convert csv to dataframe
df = pd.read_csv(
    'https://raw.githubusercontent.com/asugden/fast_ds/4db314b9c92adfb2f306c3d8ba745d32fd77ebca/data/fda_data.csv')
# Connect to database
conn = sqlite3.connect(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/sqlite_fda.db')
# Create raw table
df.to_sql('raw', conn, if_exists='replace', index=False)


create_companies = '''
CREATE TABLE IF NOT EXISTS companies (
  id INTEGER PRIMARY KEY,
  company_name VARCHAR(90) UNIQUE NOT NULL
  );
'''

create_proper = '''
CREATE TABLE IF NOT EXISTS proper_drug (
  id INTEGER PRIMARY KEY,
  drug_proper_name VARCHAR(250) UNIQUE NOT NULL,
  company_id INTEGER REFERENCES companies (id)
  );
'''

create_proprietary = '''CREATE TABLE IF NOT EXISTS proprietary_drug (
  id INTEGER PRIMARY KEY,
  drug_proprietary_name VARCHAR(100) UNIQUE NOT NULL,
  proper_drug_id INTEGER REFERENCES proper_drug (id)
  );
'''

insert_into_companies = '''
INSERT OR IGNORE INTO companies (
  company_name
)
SELECT
applicant
FROM raw;
'''

insert_into_proper = '''INSERT OR IGNORE INTO proper_drug (
  drug_proper_name,
  company_id
)
SELECT 
proper_name,
comp.id 
FROM raw
JOIN companies as comp
ON comp.company_name = CAST (raw.applicant AS VARCHAR(90));
'''

insert_into_proprietary = '''INSERT OR IGNORE INTO proprietary_drug (
  drug_proprietary_name,
  proper_drug_id
)
SELECT 
proprietary_name,
ppd.id 
FROM raw
JOIN proper_drug as ppd
ON ppd.drug_proper_name = CAST (raw.proper_name AS VARCHAR(250));
'''

queries = [create_companies, create_proper, create_proprietary,
           insert_into_companies, insert_into_proper, insert_into_proprietary]


create_rel_tables(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/sqlite_fda.db', queries)
