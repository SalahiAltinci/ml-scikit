from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

def load_data():
    train_file = '../dataset/housing_train.csv'
    data = pd.read_csv(train_file)
    data.drop('Id', axis = 1, inplace=1) # drop id column
    y = data['SalePrice'] # set y
    X = data.drop('SalePrice', axis = 1) # drop y from training data
    return X, y

def load_test_data():
    test_file = '../dataset/housing_test.csv'
    test_data = pd.read_csv(test_file)
    test_ids = test_data['Id']
    test_data.drop('Id', axis = 1, inplace=1) # drop id column
    return test_ids, test_data


def get_important_features():
    return pickle.load( open('features.pkl', 'rb'))

def filter_data(data, features):
    return data[features]

def prepare_data(data):
    im = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
    le = LabelEncoder()
    categorical = data.select_dtypes(include=[object]) # get categorical columns
    numeric = data.drop(categorical.columns, axis=1) # get numeric columns
    categorical = categorical.apply(lambda row: le.fit_transform(row)) # String values are converted into numerical values
    data = pd.concat([categorical, numeric], axis=1) # concat data again
    return im.fit_transform(data) # return data

def train(X, y):
    reg = LinearRegression(n_jobs=-1)
    reg.fit(X, y) # training
    return reg # return classifier


def main():
    X, y = load_data() # load training data
    features = get_important_features()
    X = filter_data(X, features)
    X = prepare_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    reg = train(X_train, y_train) # train data

    print reg.score(X_test, y_test)

    test_ids, test_data = load_test_data()
    test_data = filter_data(test_data, features)
    test_data = prepare_data(test_data)
    test_prediction = reg.predict(test_data)

    output = pd.DataFrame(list(zip(test_ids, test_prediction)), columns = ['Id', 'SalePrice'])

    output.to_csv('predictions.csv', index=False)

main()
#### Testing


