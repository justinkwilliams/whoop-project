import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro


def read_data(path):
    """Read in whoop data

    Args:
        path (str): location of file on disk.
        index_of_col (int, optional): Index for the index column. Defaults to none.
    Returns:
        pd.DataFrame: A DataFrame of the csv file.
    """
    df = pd.read_csv(path)
    return df


def clean_dataframe(df: pd.DataFrame, date_col: str, drop_cols: list = None, drop_na_rows: list = None, start_date: str = None, end_date: str = None):
    """ Creates a new dataframe by converting 'date' column to datetime object, adding day of the week column as category, dropping unnecessary columns, removing specific rows with missing data, adding starting and end dates and removing spaces and captial letters in column labels.

    Args:
        df (pd.DataFrame): The Dataframe to clean.
        date_col: (pd.Series): Column containg the date
        drop_cols (list, optional): List containing the column names of columns to drop. Defaults to None.
        drop_na_rows (list, optional): List containing the columns names with NaN values. Rows with NaN values in these columns will be dropped. Defaults to None.
        start_date (str, optional): The date to start the DataFrame on. Defaults to None.
        end_date (str, optional): The date to end the DataFrame on. Defaults to None.
    Returns:
        pd.Dataframe: Updated, clean dataframe.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = (df[date_col].dt.day_name()).astype('category')
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    if drop_na_rows:
        df.dropna(subset=drop_na_rows, axis=0, inplace=True)
    if start_date:
        df = df[df[date_col] >= start_date]
    if end_date:
        df = df[df[date_col] <= end_date]
    for col in df.columns:
        column_map = {col: col.lower().replace(" ", "_")}
        df = df.rename(columns=column_map)
    return df

# version 2 with indexes


def clean_dataframe_2(df: pd.DataFrame, date_col: str, drop_cols: list = None, drop_na_rows: list = None, start_date: str = None, end_date: str = None):
    df.index = pd.to_datetime(df[date_col])
    df['day_of_week'] = df.index.day_name()
    if drop_cols:
        df.drop(drop_cols, axis=1)
    if drop_na_rows:
        df.dropna(subset=drop_na_rows, axis=0)
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    for col in df.columns:
        column_map = {col: col.lower().replace(" ", "_")}
        df = df.rename(columns=column_map)
    return df


def plot_missing_data(df: pd.DataFrame):
    """Plots the number of rows with missing data for each column in a DataFrame

    Args:
        df (pd.DataFrame): Dataframe to print missing values from
    Return:
        A plot of the missing values
    """
    missing_data = df.isnull().sum()
    fig, ax = plt.subplots()
    ax.bar(missing_data.index, missing_data.values)
    ax.set_title('Number of Rows with Missing Data')
    ax.set_xlabel('Column Name')
    ax.set_ylabel('Number of Rows')
    plt.xticks(rotation=90)
    plt.show()


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


def nap_bool(df: pd.DataFrame, nap_col: str):
    """Converts the nap column to a boolean and changes column name to "nap".

    Args:
        df (pd.DataFrame): DataFrame with nap column.
        nap_col (str): Name of the nap column.
    Returns:
        pd.DataFrame: DataFrame with new nap column.
    """
    df['nap'] = df['nap_count'].fillna(0)
    df['nap'] = (df['nap'] >= 1).astype(bool)
    return df


def recovery_cat(df: pd.DataFrame, recov_col: str):
    """"Creates a new column 'recovery_level_cat' with recovery as a category.

    Args:
        df (pd.DataFrame): DataFrame with recovery column.
        recov_col (str): Name of recovery column.
    Returns:
        pd.DataFrame: DataFrame with new recovery column.
    """
    df['recovery_level_cat'] = pd.cut(df[recov_col], bins=[
        0, 33, 66, 100], labels=['red', 'yellow', 'green']).astype('category')
    return df
