import re
import string
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.metrics
from sklearn.model_selection import train_test_split






df = pandas.read_csv("census-income.csv")
print(df)
df.dropna()
print(df)
# print(df)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
classes = df.iloc[:,-1]
instances = df.iloc[:,:-1]
print(instances)
print(classes)


file = open("Census Feature Descriptions.txt")
for row in file:
    print(row)


# mapping = {}
# for line in file:
#     line = re.sub('\t+','=', line)
#     (key, val) = line.split("=")
#     mapping[(key.strip('\n'))] = val
#
#
# mapping = d_swap = {v: k for k, v in mapping.items()}
#
#
# for item in df.columns:
#     temp = item
#     temp = temp + '\n'
#     if temp in mapping.keys():
#         df = df.rename(columns={temp: mapping[temp]})




