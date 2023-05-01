import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro


def read_data(path):
    """Read in whoop data

    Args:
        path (str): location of file on disk
    """
    df = pd.read_csv(path)
    return df


def clean_dataframe(df: pd.DataFrame, date_col: pd.Series, drop_cols: list, drop_na_rows: list, start_date: str, end_date: str):
    """ Creates a new dataframe by converting 'date' column to datetime object, adding day of the week column as category, dropping unnecessary columns, removing specific rows with missing data, adding starting and end dates and removing spaces and captial letters in column labels.

    Args:
        df (pd.DataFrame): The Dataframe to clean.
        date_column: (pd.Series): Column containg the date
        drop_col (list): List containing the column names of columns to drop.
        drop_na_rows (list): List containing the columns names with NaN values. Rows with NaN values in these columns will be dropped.
        start_date (str): The date to start the DataFrame on
        end_date (str): The date to end the DataFrame on
    Returns:
        pd.Dataframe: Updated, clean dataframe.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.day_name()
    df_clean = df.loc[(df[date_col] >= start_date)
                      & (df[date_col] < end_date)]
    df_clean = df.drop(drop_cols, axis=1).dropna(subset=drop_na_rows, axis=0)
    for col in df_clean.columns:
        column_map = {col: col.lower().replace(" ", "_")}
        df_clean = df_clean.rename(columns=column_map)
    return df_clean


def fill_missing_values(df: pd.DataFrame, col_list: list):
    """"Fills missing values of a column with the mean in normally distributed determined by using Shapiro-Wilk test.

    Args:
        df (pd.DataFrame): DataFrame with columns.
        col_list (list): List of columns with missing data.

    Returns:
        pd.DataFrame: Modified DataFrame with filled missing values.
    """
    for col in col_list:
        stat, p = shapiro(df[col].dropna())
        if p > 0.05:
            df[col] = df[col].fillna(df[col].mean())
        else:
            print(col + " is not normally distributed")
    return df
