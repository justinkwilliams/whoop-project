import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, features: list[str], labels: str, seed: int, train_fraction: float = 0.7):
    """"
    Splits a DataFrame into training and testing sets.

    Args:
        data (pd.DataFrame): DataFrame that is to be split
        features (list[str]): List of column names of features to be used in model
        labels (str): Column name containing target variable 
        seed (int): Random integer to set as random state
        train_fraction (float, optional): The fraction of the data used to train the model. Defaults to 0.7.

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


def train_model(train_features: pd.DataFrame, train_labels: pd.Series, params: dict = None):
    """
    Trains a XGBoost Classifier on training features and labels with specific paramaters.

    Args:
        train_features (pd.DataFrame | np.ndarray): Set of training features
        train_labels (pd.Series | np.ndarray): Set of training labels
        params (dict, optional):A dictionary of parameters and their values to pass to the model. Defaults to None.

    Returns:
       xgboost.XGBClassifier: Trained XGBoost Classifier
    """
    xgb_model = xgb.XGBRegressor(params)
    xgb_model.fit(train_features, train_labels)
    return xgb_model


def predict(model: xgb.XGBRegressor, test_features: pd.DataFrame):
    """
    Perform predictions on an XGBRegressor given testing features and a specific threshold.

    Args:
        model (xgboost.XGBClassifier): Model to perform predictions on
        test_features (pd.DataFrame): Set of testing features
    Returns:
        y_pred (np.ndarray): Contains an array of predicted class labels.

     """
    y_pred = model.predict(test_features)
    return y_pred


def assess(model: xgb.XGBClassifier, test_features: pd.DataFrame, test_labels: pd.Series):
    """Asses the the model by calculating mean squared error and R**2

    Args:
        model (xgboost.XGBClassifier): Model to perform assesments on
        test_features (pd.DataFrame): Set of testing features
        test_labels (pd.Series): Set of testing labels

    Returns:
        dict: A dictionary containing the accuracy score, f1-score, precision score, recall score, and roc-auc score for the model
    """
    y_pred = predict(model, test_features)
    mse = mean_squared_error(test_labels, y_pred)
    r2 = r2_score(test_labels, y_pred)
    return {
        'mse': mse,
        'r2': r2
    }
