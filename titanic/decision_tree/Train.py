"""
    Decision Tree implementation for Titanic survival data
    Implemented for data visualisation
"""
import sys
sys.path.append("../../helpers/")
import preprocessor

from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer


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


def main():
    train_X, train_y = preprocessor.load_csv('../../dataset/titanic_data.csv', 'Survived') # load training data

    features = train_X.columns.values

    train_X = prepare_data(train_X)
    clf = tree.DecisionTreeClassifier(min_samples_split=50, max_leaf_nodes=10)
    clf = clf.fit(train_X, train_y)     # Train the data

    visualise(clf, features)


def visualise(clf, features):

    with open("tree.dot","w") as f:
        f = tree.export_graphviz(clf,
                                 feature_names=features,
                                 class_names=["Not Survived","Survived"],
                                 out_file=f)
    # dot -Tpdf tree.dot -o tree.pdf


if __name__ == '__main__':
    main()