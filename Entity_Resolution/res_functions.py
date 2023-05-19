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


bins = np.arange(0, 1.1, 0.1)

bins[-1] = 1

bin_labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
              '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1']

df['bin'] = pd.cut(df[col_name], bins=bins, labels=bin_labels)

sampled_df = pd.DataFrame()
