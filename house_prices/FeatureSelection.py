
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle

def load_data():
    train_file = '../dataset/housing_train.csv'
    data = pd.read_csv(train_file)
    data.drop('Id', axis = 1, inplace=1) # drop id column
    y = data['SalePrice'] # set y
    X = data.drop('SalePrice', axis = 1) # drop y from training data
    return X, y

def prepare_data(data):
    im = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
    le = LabelEncoder()
    # target
    categorical = data.select_dtypes(include=[object]) # get categorical columns
    numeric = data.drop(categorical.columns, axis=1) # get numeric columns
    categorical = categorical.apply(lambda row: le.fit_transform(row)) # String values are converted into numerical values
    data = pd.concat([categorical, numeric], axis=1) # concat data again
    columns = data.columns.values # get column names
    return im.fit_transform(data), columns # return data and columns

def train(X, y):
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X, y) # training
    return clf # return classifier

def get_feature_importances(clf, columns):
    features = pd.Series(clf.feature_importances_, index=columns) # convert array to Pandas Series
    features.sort_values(ascending=False, inplace=True) # sort by descending
    features = features.apply(lambda x: x*1000) # rescale values
    return features[features > 0] # eliminate unimportant values and return

def get_statistics(features):
    return {
        'mean': np.mean(features),  # mean of distribution
        'std' : np.std(features),  # standard deviation of distribution
        'median' : np.median(features), # median
        'min' : np.min(features), # minimum
        'max' : np.max(features) # maximum
    }

def export_features(feature_importances, threshold):
    features = feature_importances[feature_importances > threshold].index.values # get keys
    pickle.dump(features,open('features.pkl','wb')) # export

def main():
    X, y = load_data() # load training data
    X, columns = prepare_data(X)
    clf = train(X, y) # train data
    feature_importances = get_feature_importances(clf, columns) # Calculate feature importances
    statistics = get_statistics(feature_importances) # get statistics
    feature_importances.to_csv('feature_importances.csv') # export features to a csv file
    export_features(feature_importances, 8) # export features to use in regression

main()