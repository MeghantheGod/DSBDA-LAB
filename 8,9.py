import pandas as pd

df = pd.read_csv("/home/meghan/Downloads/titanic.csv")
df
df.isnull().sum()
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked']
df['Age'] = df['Age'].fillna(df.Age.mean())
df['Age']
a = df['Age'].mean()
a
b=int(df.Age.mean())
b
df['NewCabin'] = df['Cabin'].fillna(0)
df['NewCabin'] = df['Cabin'].notnull().astype(int)
df['NewCabin'] 

df.drop(columns=['Cabin'], inplace=True)
df
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
survived = df['Survived'].value_counts()
survived
plt.figure(figsize = (8,6))
plt.pie(survived, labels=['Not Survived', 'Survived'],
        autopct = '%1.1f%%', colors = ['Red', 'Blue'])
plt.title('Survived vs Not Survived')
plt.legend()
plt.show()
plt.figure(figsize = (8,6))
sns.countplot(x=survived, hue = 'Survived', data = df, palette = ['Red', 'Blue'], legend=False)
plt.title('Survival Status')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
survival_by_Pclass = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
survival_by_Pclass
survival_by_Pclass.plot(kind = 'bar', color = ['Orange', 'Green'])
plt.title("Survival by Passenger Class")
plt.ylabel("Survived")
plt.xlabel("Passenger Class")
plt.show()
plt.figure(figsize = (8,6))
sns.histplot(df['Fare'], bins = 20, kde = True)
plt.title('Distribution Of Ticket Fares')
plt.xlabel('Fares')
plt.ylabel('Frequency')
plt.show()
sns.boxplot(x='Sex', y='Age', data = df, hue = 'Survived')
sns.violinplot(x='Sex', y='Age', data = df, hue = 'Survived', palette= ['Red', 'Green'])
