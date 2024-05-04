import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris(as_frame = True)
X = iris['data']
Y = iris['target']
X
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 99)
X_train
Y_train
X.isnull().sum()
Y.isnull().sum()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy is {accuracy}")
precision = precision_score(Y_test, y_pred, average = 'macro')
print(f'Precision is {precision}')
recall = recall_score(Y_test, y_pred, average = 'macro')
print(f'Recall is {recall}')
classification = classification_report(Y_test, y_pred)
print(f' {classification}')
conf_matrix = confusion_matrix(Y_test, y_pred)
print(f'Confusion matrix is {conf_matrix}')
# VALIDATION
data = [[1,4,5,6,3,5,4,3],[3,4,2,6,3,8,4,9],[4,5,7,9,8,7,4,5]]
std = StandardScaler()
new_data = std.fit_transform(data)
new_data
data_pred = model.predict(data)

data_pred
