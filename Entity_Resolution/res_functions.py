import jarowinkler
import numpy as np
import pandas as pd
import xgboost
from thefuzz import fuzz


def set_overlap(a: str, b: str, com_parts: list) -> float:
    a_new = a.replace(".", "")
    b_new = b.replace(".", "")
    for part in com_parts:
        a_new = a_new.lower().replace(part.lower(), "").strip()
        b_new = b_new.lower().replace(part.lower(), "").strip()
    set1 = set(a_new.lower().split(' '))
    set2 = set(b_new.lower().split(' '))
    overlap = len(set1.intersection(set2))/min(len(set1), len(set2))
    return overlap


def clean_df(df: pd.DataFrame, column: str, com_parts: list) -> pd.Series:
    clean_column = column + '_clean'
    df[clean_column] = df[column]
    for part in com_parts:
        df[clean_column] = df[clean_column].str.replace(
            part, '', case=False).str.strip()
    df[clean_column] = df[clean_column].str.replace(".", "")
    df[clean_column] = df[clean_column].str.lower()
    return df


def jaro_winkler_similarity(str1, str2) -> float:
    a_new = a.replace(".", "")
    b_new = b.replace(".", "")
    for part in com_parts:
        a_new = a_new.lower().replace(part.lower(), "").strip()
        b_new = b_new.lower().replace(part.lower(), "").strip()
    similarity = jarowinkler.jaro_similarity(a_new, b_new)
    return similarity


def fuzz_partial_ratio(a: str, b: str, com_parts: list) -> float:
    a_new = a.replace(".", "")
    b_new = b.replace(".", "")
    for part in com_parts:
        a_new = a_new.lower().replace(part.lower(), "").strip()
        b_new = b_new.lower().replace(part.lower(), "").strip()
    partial_ratio = fuzz.partial_ratio(a_new, b_new)
    return partial_ratio
