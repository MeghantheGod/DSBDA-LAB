import pandas as pd
df = pd.read_csv('/home/meghan/Downloads/dirtydata.csv')
df
df.head()
df.describe()
df.shape
features_with_nan = [feat for feat in df.columns if df[feat].isnull().sum() > 0 and df[feat].dtype != '0']
for feat in df.columns:
        print('{} has {} % missing values'.format(feat, df[feat].isnull().mean()))
features_with_nan
for feat in features_with_nan:
    mean_value = df[feat].isnull().mean()
    df[feat] = df[feat].fillna(mean_value)
df[features_with_nan].isnull().sum()
for feat in df.columns:
    print('{} has {}  data type'.format(feat,df[feat].dtypes))
df['Calories'] = df['Calories'].astype('int64')
for feat in df.columns:
    print('{} has {}  data type'.format(feat,df[feat].dtypes))
import numpy as np
df['Calories'] = np.where((df['Calories'] < 0), -(df['Calories']), df['Calories'])
(df['Calories']<0).value_counts()
df.describe()
