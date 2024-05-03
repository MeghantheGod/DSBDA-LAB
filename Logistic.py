import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/Social_Networking_Ads.csv")
df

numerical_feat = [feat for feat in df.columns if df[feat].dtypes != '0' and feat not in ['User ID']]
numerical_feat
df[numerical_feat].isnull().sum()
import seaborn as sns
sns.kdeplot(x='Age', data = df)
sns.kdeplot(x='EstimatedSalary' , data = df)
sns.boxplot(x='Age', data = df)
sns.boxplot(x='EstimatedSalary', data = df)
sns.barplot(x='Gender', y='Age', data = df)
sns.barplot(x='Gender', y = 'EstimatedSalary', data = df)
sns.scatterplot(x='Age', y='EstimatedSalary', data = df)
df.drop(['User ID'], axis = 1, inplace = True)
df
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df.drop('Purchased', axis = 1), df['Purchased'], test_size = 0.3, random_state = 0)
X_train
Y_train
one_hot_encoded_data = pd.get_dummies(X_train, columns = ['Gender'])
X_train_enc = pd.DataFrame(one_hot_encoded_data)
X_train_enc
one_hot_encoded_data_Y = pd.get_dummies(X_test, columns = ['Gender'])
X_test_enc = pd.DataFrame(one_hot_encoded_data_Y)
X_test_enc
X_train_enc.drop('Gender_Male', axis = 1, inplace = True)
X_train_enc.head()
X_test_enc.drop('Gender_Male', axis = 1, inplace = True)
X_test_enc
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_enc)
X_train_scaled = scaler.transform(X_train_enc)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train_enc.columns)
X_train_scaled
X_test_scaled = scaler.transform(X_test_enc)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test_enc.columns)
X_test_scaled
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train_scaled, Y_train)
predicted_values = clf.predict(X_test_scaled)
predicted_values.mean()
from sklearn.metrics import (
confusion_matrix,
accuracy_score,
precision_score, 
recall_score,
f1_score)
conf_matrix = confusion_matrix(Y_test, predicted_values)
conf_matrix
accuracy = accuracy_score(Y_test, predicted_values)
precision = precision_score(Y_test, predicted_values)
recall = recall_score(Y_test, predicted_values)
f1_score = f1_score(Y_test, predicted_values)
print(f"Accuracy is {accuracy}")
print(f"Recall is {recall}")
print(f"F-1 Score is {f1_score}")
import matplotlib.pyplot as plt

ax = sns.heatmap(
    conf_matrix, 
    annot = True,
    fmt = 'd',
    cbar = False,
    cmap = 'flag',
    vmax = 175)
ax.set_xlabel('Predicted', labelpad = 20)
ax.set_ylabel('Actual', labelpad = 20)
plt.show()
