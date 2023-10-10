import re
import string
import pandas
import numpy
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold


# columns = []
# file = open("Census Feature Descriptions.txt")
# for row in file:
#     columns.append(row)


# Read the CSV file
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
df = pandas.read_csv("census-income.csv")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#get rid of ?
df.replace("?", numpy.nan, inplace=True)
df.dropna(inplace=True)

#Reset the index after dropping rows
df = df.reset_index(drop=True)

# print(df)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
classes = df.iloc[:, -1]
instances = df.iloc[:, :-1]

# print(instances)
# print(classes)

# Identify columns with non-numeric data
categorical_columns = instances.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
for column in categorical_columns:
    instances = pandas.get_dummies(instances, columns=[column], prefix=column)

X_train, X_test, y_train, y_test = train_test_split(instances, classes, test_size=0.2, random_state=42)

#DECISION TREE

dt = DecisionTreeClassifier(max_depth=14)
dt.fit(X_train,y_train)
dtprediction = dt.predict(X_test)
print(dtprediction)

print("\n DATA: Decision Tree")
print("classification report:")

print(classification_report(y_test, dtprediction))

print("accuracy score: ")
print(accuracy_score(y_test, dtprediction))

print("confusion matrix:")
print(confusion_matrix(y_test, dtprediction))

cross_val_scores = cross_val_score(dt, X_train, y_train, cv=kf, scoring='accuracy')

print("Cross-Validation Scores:", cross_val_scores)

print("Average Accuracy:", cross_val_scores.mean())


#RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100, max_depth=14)
rf.fit(X_train,y_train)
rfprediction = rf.predict(X_test)

print(rfprediction)

print("\n DATA: Random Forest")
print("classification report:")

print(classification_report(y_test, rfprediction))

print("accuracy score: ")
print(accuracy_score(y_test, rfprediction))

print("confusion matrix:")
print(confusion_matrix(y_test, rfprediction))

cross_val_scores = cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy')

print("Cross-Validation Scores:", cross_val_scores)

print("Average Accuracy:", cross_val_scores.mean())


df2 = pandas.read_csv("census-income-test.csv")
test_prediction = rf.predict(df2)


with open("Weglinski_Patrick.txt", 'w') as file:
    for i in range(len(test_prediction)):
        file.write(str(test_prediction[i]))
        file.write("\n")






