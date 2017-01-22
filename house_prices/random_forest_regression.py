# Linear regression for a Kaggle competition
#
import sys
sys.path.append("../helpers/")
import preprocessor

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
import pickle
import os.path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def grid_search(estimator, param_grid):
    return GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=5, verbose=5)


def get_important_features():
    """ Load pre-selected important features
    """
    if os.path.isfile('features.pkl'):
        return pickle.load( open('features.pkl', 'rb'))
    else:
        return None


def filter_data(data, features):
    """
    """
    return data[features]


def prepare_data(data):
    """ Prepares data for the algorithm.
    Fills NaN values, and converts categorical features into numerical features.
    """
    im = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
    le = LabelEncoder()
    categorical = data.select_dtypes(include=[object]) # get categorical columns
    numeric = data.select_dtypes(exclude=[object]) # get categorical columns
    categorical = categorical.apply(lambda row: le.fit_transform(row)) # String values are converted into numerical values
    data = pd.concat([categorical, numeric], axis=1) # concat data again
    return im.fit_transform(data) # return data


def train(X, y):
    rf_reg = RandomForestRegressor()

    param_grid={
        'n_estimators':[300],
        'max_depth':[9],
        'max_leaf_nodes':[85],
        'random_state' : [1]
    }
    reg = grid_search(rf_reg, param_grid)
    reg.fit(X, y) # training
    return reg # return classifier


def test(regressor, features=None):
    """ Test the model with unseen data
    """
    test_X, test_ids = preprocessor.load_csv('../dataset/housing_test.csv', 'Id') # load training data

    if features is not None:
        test_X = filter_data(test_X, features)

    test_X = prepare_data(test_X)
    test_prediction = regressor.predict(test_X)
    export_results(test_ids, test_prediction)


def export_results(test_ids, test_prediction):
    """ Export results for the Kaggle competition
    """
    output_linear_reg = pd.DataFrame(list(zip(test_ids, test_prediction)), columns = ['Id', 'SalePrice'])
    output_linear_reg.to_csv('output_linear_reg.csv', index=False)

def main():
    """
    """
    X, y = preprocessor.load_csv('../dataset/housing_train.csv', 'SalePrice', ['Id']) # load training data
    features = get_important_features()

    if features is not None:
        X = filter_data(X, features)

    X = prepare_data(X)

    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    linear_reg = train(X_train, y_train) # train data
    print 'Linear Regression score for training: ', linear_reg.score(X_test, y_test)

    test(linear_reg, features) # start testing

if __name__ == '__main__':
    main()

