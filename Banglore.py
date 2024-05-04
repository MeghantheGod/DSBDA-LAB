import pandas as pd
df = pd.read_csv('/home/meghan/Downloads/Banglore Housing Prices.csv')
df
df.info()
df.describe()

df.head()
df.isnull().sum()
df.dropna(inplace = True)
df.isnull().sum()
df['size'] = [int(value.split(' ')[0])for value in df['size']]
df.head()
df['total_sqft'].head()
df['total_sqft'].info()
def convert_sqft(value):
    # if value is in range
 try:
    if ('-') in value:
        start, end = map(float, value.split('-'))
        return (start + end)/2
    else:
        return float(value)
 except ValueError:
    return float('nan')
df['total_sqft'] = [convert_sqft(value) for value in df['total_sqft']]
df['total_sqft'].head()
df['total_sqft']
df['total_sqft'].isnull().sum()
df.isnull().sum()
df.dropna(inplace = True)
df['total_sqft']
df['total_sqft'].isnull().sum()
df['Price_per_sqft'] = df['price'] / df['total_sqft']
df['Price_per_sqft'].head()
df.head()
selected_columns = ['Price_per_sqft','size']
outliers = df[selected_columns]
import seaborn as sns
import matplotlib.pyplot as plt
for i in outliers:
    sns.boxplot(x=df[i])
    plt.show()
def rem_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upr_bound = Q3 + (1.5 * IQR)
    return column[(column >= lower_bound) and (column <= upr_bound)]
import numpy as np
df['Price_per_sqft'] = rem_outliers(df['Price_per_sqft'])
df['size'] = rem_outliers(df['size'])
sns.boxplot(x=df['size'])
plt.show()
sns.boxplot(x=df['Price_per_sqft'])
plt.show()
df.isnull().sum()
from sklearn.model_selection import train_test_split, cross_val_score
df.head()
X = df[['size','total_sqft','bath','Price_per_sqft']]
Y = df['price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error is {mse}")
r2score = r2_score(Y_test, y_pred)
print(f"R-2 Score is {r2score}")
cv = np.mean(cross_val_score(model, X, Y, cv=5))
print(f"Cross Value Score is {cv}")
