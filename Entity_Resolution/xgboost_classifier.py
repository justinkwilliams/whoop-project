import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    """
    Read in data from a CSV file.
    Data type specifc

    Args:
        path (str): the location of the file on your computer

    Returns:
        pd.DataFrame: the data, loaded in and possibly cleaned

    """
    df = pd.read_csv(path)
    return df


def split_data(data: pd.DataFrame, features: list[str], labels: str, train_fraction: float = 0.7, seed: int = None):
    """"
    Splits a DataFrame into training and testing sets.

    Args:
        data (pd.DataFrame): DataFrame that is to be split
        features (list[str]): List of column names of features to be used in model
        labels (str): Column name containing target variable 
        train_fraction (float, optional): The fraction of the data used to train the model. Defaults to 0.7.
        seed (int) : An integer value to use as the random_state number. Defaults to None

    Returns:
        tuple: A tuple containing training and testing sets of features and labels:
            X_train (pd.DataFrame): Set of training features
            X_test (pd.DataFrame): Set of testing features
            y_train (pd.Series): Set of training labels
            y_test (pd.Series): Set of testing labels

     """
    y = data[labels]
    X = data[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_fraction, random_state=seed)
    return X_train, X_test, y_train, y_test


def train_model(train_features: pd.DataFrame | np.ndarray, train_labels: pd.Series | np.ndarray, seed: int = None):
    """
    Trains a XGBoost Classifier on training features and labels with specific paramaters.

    Args:
        train_features (pd.DataFrame | np.ndarray): Set of training features
        train_labels (pd.Series | np.ndarray): Set of training labels
        seed (int) : An integer value to use as the random_state number. Defaults to None

    Returns:
       xgboost.XGBClassifier: Trained XGBoost Classifier

    """
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic", random_state=seed)
    xgb_model.fit(train_features, train_labels)
    return xgb_model


def predict(model: xgb.XGBClassifier, test_features: pd.DataFrame, threshold: float):
    """
    Perform predictions on an XGBClassifier given testing features and a specific threshold.

    Args:
        model (xgboost.XGBClassifier): Model to perform predictions on
        test_features (pd.DataFrame): Set of testing features
        threshold (float): Value to compare postive class to in order to perform predictions

    Returns:
        tuple: Tuple of two numpy arrays containing predict class labels and probabilities:
            y_pred (np.ndarray): Contains an array of predicted class labels
            y_pred_proba (np.ndarray) : Contains an array of predicted class probabilities for the postive class

     """
    y_pred = model.predict(test_features)
    y_pred_proba = model.predict_proba(test_features)[:, 1] >= threshold
    return y_pred, y_pred_proba


def predict_proba(model: xgb.XGBClassifier, test_features: pd.DataFrame):
    """"
        Performs probability predictions on an XGBClassifier given testing features.

    Args:
        model (xgboost.XGBClassifier): Model to perform predictions on
        test_features (pd.DataFrame): Set of testing features

    Returns:
        y_pred_proba (np.ndarray) : Contains an array of predicted class probabilities for the postive class
"""
    y_pred_proba = model.predict_proba(test_features)[:, 1]
    return y_pred_proba


def assess(model: xgb.XGBClassifier, test_features: pd.DataFrame, test_labels: pd.Series, threshold: float = 0.5):
    """Asses the accuracy, f1-score, precision, recall, and roc-auc-score of an XGBClassifier

    Args:
        model (xgboost.XGBClassifier): Model to perform assesments on
        test_features (pd.DataFrame): Set of testing features
        test_labels (pd.Series): Set of testing labels

    Returns:
        dict: A dictionary containing the accuracy score, f1-score, precision score, recall score, and roc-auc score for the model
    """
    y_pred, y_pred_proba = predict(model, test_features, threshold)
    accuracy = accuracy_score(test_labels, y_pred)
    f1_scoree = f1_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_features, y_pred)
    roc_auc = roc_auc_score(test_features, y_pred_proba)
    return {
        'accuracy': accuracy,
        'f1': f1_scoree,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


def assess_part(model: xgb.XGBClassifier, test_features: pd.DataFrame, test_labels: pd.Series, threshold: float = 0.5):
    """Asses the accuracy, f1-score, precision, recall, and roc-auc-score of an XGBClassifier

    Args:
        model (xgboost.XGBClassifier): Model to perform assesments on
        test_features (pd.DataFrame): Set of testing features
        test_labels (pd.Series): Set of testing labels

    Returns:
        dict: A dictionary containing the accuracy score, f1-score, precision score, recall score, and roc-auc score for the model
    """
    y_pred, y_pred_proba = predict(model, test_features, threshold)
    accuracy = accuracy_score(test_labels, y_pred)
    f1_scoree = f1_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    return {
        'accuracy': accuracy,
        'f1': f1_scoree,
        'precision': precision
    }
