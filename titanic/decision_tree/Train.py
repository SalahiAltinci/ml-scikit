
"""
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.

"""

from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_file = 'titanic_data.csv'

data = pd.read_csv(train_file)

feautures = ['Sex', 'Age', 'Pclass','Fare']
outcome_feature = ['Survived']
full_features = feautures + outcome_feature
#

data = data[full_features]
data = data.dropna()       # removing rows with missing data

le = LabelEncoder()
data = data.apply(lambda row: le.fit_transform(row))    # String values are converted into numerical values

x = data[feautures]
y = data['Survived']

clf = tree.DecisionTreeClassifier(min_samples_split = 50, max_leaf_nodes = 10)
clf = clf.fit(x, y)     # Train the data

with open("tree.dot","w") as f:
    f = tree.export_graphviz(clf,
                             feature_names=feautures,
                             class_names=["Not Survived","Survived"],
                             out_file=f)

# dot -Tpdf result.dot -o x.pdf
