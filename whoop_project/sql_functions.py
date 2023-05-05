import sqlite3

import pandas as pd
import sqlalchemy


def create_raw_table(path_to_csv: str, path_to_db: str, table_name: str, dtypes: dict = None):
    """"Converts a csv to a DataFrame and then converts the DataFrame to a sql table.

    Args:
        path_to_csv (str): File path to csv file to convert.
        path_to_db (str): File path to database to add table.
        table_name (str): Name of created table.
        dtypes (dict, optional): Dictionary filled with column names as keys and desired sql datatypes as values. Defaults to None.
    """
    df = pd.read_csv(path_to_csv, dtype=dtypes)
    engine = sqlalchemy.create_engine(f"sqlite:///{path_to_db}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)


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
