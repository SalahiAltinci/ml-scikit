# Helpers module

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(file_path=None, target=None, unwanted_columns=None):

    X = pd.read_csv(file_path)  # load the file

    if target is not None:
        y = X[target]  # set y
        X.drop(target, axis=1, inplace=1)  # drop y from training data
    else:
        y = None

    if unwanted_columns is not None:
        X.drop(unwanted_columns, axis=1, inplace=1)  # drop unwanted columns


    return X, y

def split_data(X, y, _test_size=0.20, _random_state=1):
     return train_test_split(X, y, test_size=_test_size, random_state=_random_state)