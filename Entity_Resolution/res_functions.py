import jellyfish
import numpy as np
import pandas as pd
import xgboost
from thefuzz import fuzz


def set_overlap_df(df: pd.DataFrame, column1: str, column2: str, com_parts: list = None) -> pd.DataFrame:
    overlap_values = []
    for index, row in df.iterrows():
        a_new = str(row[column1]).replace(".", "")
        b_new = str(row[column2]).replace(".", "")
        if com_parts:
            for part in com_parts:
                a_new = a_new.lower().replace(part.lower(), "").strip()
                b_new = b_new.lower().replace(part.lower(), "").strip()
        set1 = set(a_new.lower().split(' '))
        set2 = set(b_new.lower().split(' '))
        overlap = len(set1.intersection(set2)) / min(len(set1), len(set2))
        overlap_values.append(overlap)
    df['set_overlap'] = overlap_values
    return df


def jaro_winkler_similarity_df(df: pd.DataFrame, column1: str, column2: str, com_parts: list = None) -> pd.DataFrame:
    similarity_values = []
    for index, row in df.iterrows():
        str1 = str(row[column1])
        str2 = str(row[column2])
        a_new = str1.replace(".", "")
        b_new = str2.replace(".", "")
        if com_parts:
            for part in com_parts:
                a_new = a_new.lower().replace(part.lower(), "").strip()
                b_new = b_new.lower().replace(part.lower(), "").strip()
        similarity = jellyfish.jaro_winkler(a_new, b_new)
        rounded_similarity = round(similarity, 3)
        similarity_values.append(rounded_similarity)
    df['jaro_score'] = similarity_values
    return df


def fuzz_partial_ratio_df(df: pd.DataFrame, column1: str, column2: str, com_parts: list = None) -> pd.DataFrame:
    partial_score = []
    for index, row in df.iterrows():
        str1 = str(row[column1])
        str2 = str(row[column2])
        a_new = str1.replace(".", "")
        b_new = str2.replace(".", "")
        if com_parts:
            for part in com_parts:
                a_new = a_new.lower().replace(part.lower(), "").strip()
                b_new = b_new.lower().replace(part.lower(), "").strip()
        partial_values = fuzz.partial_ratio(a_new, b_new)
        partial_score.append(partial_values)
    df['part_ratio'] = partial_score
    return df


def df_with_features(df: pd.DataFrame, column1: str, column2: str, com_parts: list = None):
    df = set_overlap_df(df, column1, column2, com_parts)
    df = jaro_winkler_similarity_df(df, column1, column2, com_parts)
    df = fuzz_partial_ratio_df(df, column1, column2, com_parts)
    return df


def select_bins(df: pd.DataFrame,
                col: str,
                n_samples: int,
                bin_width: float = 0.1) -> pd.DataFrame:
    """Take a dataframe and split one column into bins
    and sample from them

    Args:
        df (pd.DataFrame): input dataframe
        col (str): the name of the dataframe column
        n_samples (int): number of samples per bin
        bin_width (float, optional): size of the bin. 
            Defaults to 0.1.

    Returns:
        pd.DataFrame: the set of samples
    """
    out = []
    bins = np.array(range(0, 1, bin_width))
    for binmin in bins:
        # Fencepost problem
        binmax = binmin + bin_width
        subdf = df.loc[df[col] >= binmin & df[col] < binmax]
        out.append(subdf.sample(n_samples, replace=False))
    return pd.concat(out)


def rows_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Takes in raw data and converts it into pairs of entries
    for comparison

    Args:
        df (pd.DataFrame): raw list of rows

    Returns:
        pd.DataFrame: set of pairs

    """
    pairs = []
    for i, row1 in df.iterrows():
        # An underscore is a variable that exists
        # but will never be used again
        for _, row2 in df.iloc[i+1:].iterrows():
            pairs.append([row1['id'], row1['company_name'],
                         row2['id'], row2['company_name']])
    return pd.DataFrame(pairs, columns=['id1', 'name1', 'id2', 'name2'])
