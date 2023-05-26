import pandas as pd
import sqlalchemy

# Create sqlalchemy engine
engine = sqlalchemy.create_engine(
    'sqlite:////Users/user/Desktop/Real Whoop Project/whoop_project/Data/fda_sqlalchemy.db')

# Read in csv to DataFrame
df = pd.read_csv(
    '/Users/user/Desktop/Real Whoop Project/whoop_project/Data/id_pairs.csv', usecols=['id1', 'id2'])

# Convert DataFrame to table
df.to_sql('company_pairs_raw', engine, index=False, if_exists='replace')

# Create company_pairs table
engine.execute('CREATE TABLE company_pairs (company_id1 INT NOT NULL, company_id2 INT NOT NULL, FOREIGN KEY (id1) REFERENCES companies (id), FOREIGN KEY (id2) REFERENCES companies (id))')

# Insert data into company_pairs table
engine.execute(
    'INSERT INTO company_pairs (company_id1, company_id2) SELECT id1, id2 FROM company_pairs_raw')

# Drop company_pairs_raw
engine.execute('DROP TABLE company_pairs_raw')
