import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/Iris.csv")
df
df.isnull().sum()
import seaborn as sns
sns.histplot(x='SepalLengthCm', hue = 'Species', data = df)
sns.histplot(x='SepalWidthCm', hue = 'Species', data = df)
sns.histplot(x='PetalLengthCm', hue = 'Species', data = df)
sns.histplot(x='PetalWidthCm', hue = 'Species', data = df)
import matplotlib.pyplot as plt
arr = ['SepalLengthCm','SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
plt.boxplot(df[arr])
plt.xlabel(arr)
plt.show()
sns.boxplot(x='Species', y='SepalLengthCm', data = df)
sns.boxplot(x='Species', y='SepalWidthCm', data = df)
sns.boxplot(x='Species', y='PetalLengthCm', data = df)
sns.boxplot(x='Species', y='PetalWidthCm', data = df)
