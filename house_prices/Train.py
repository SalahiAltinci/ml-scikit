
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

sel = VarianceThreshold(threshold=0.50)
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
le = LabelEncoder()

train_file = '../dataset/housing_train.csv'

data = pd.read_csv(train_file)

prices = data['SalePrice']
data = data.drop('SalePrice', axis = 1)


columns = pd.DataFrame(data.columns.values)


data = data.apply(lambda row: le.fit_transform(row))    # String values are converted into numerical values
data = im.fit_transform(data)
data = sel.fit_transform(data)

columns = columns[sel.get_support()]

X_train, X_test, y_train, y_test = train_test_split(data, prices, test_size=0.30, random_state=10)

reg = LinearRegression(normalize=True,n_jobs=4)
reg.fit(X_train, y_train)

prediction = reg.predict(X_test)

print r2_score(y_test, prediction)


# Plot outputs
plt.scatter(X_test['SalePrice'], y_test, color='black')
plt.plot(X_test['SalePrice'], prediction, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())

plt.show()

#### Testing

test_file = '../dataset/housing_test.csv'
test_data = pd.read_csv(test_file)

test_ids = test_data['Id']

test_data = test_data.apply(lambda row: le.fit_transform(row))    # String values are converted into numerical values

test_columns = columns.transpose().values[0]

test_data = test_data[test_columns]

test_prediction = reg.predict(test_data)

output = pd.DataFrame(list(zip(test_ids, test_prediction)), columns = ['Id', 'SalePrice'])
output.to_csv('predictions.csv', index=False)