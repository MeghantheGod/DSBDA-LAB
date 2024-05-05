import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/Iris.csv")
df
df.drop("Id", axis = 1, inplace = True)
df
categories = [i for i in df['Species'].unique()]
categories
features = [feat for feat in df.columns if df[feat].dtypes != '0']
features
df.isnull().sum()
species_group = df.groupby('Species')
import statistics as st
species_group.mean()
species_group.std()
species_group.var()
pd.options.display.max_columns = 100
species_group.describe()
# Finding Covariance and Correlation
def covariance(x,y):
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    num = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    denom = len(x) - 1
    cov = num/denom
    return cov
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        if(i<j and i != j):
            val = covariance(df[features[i]], df[features[j]])
            print("Covariance for {} and {}: {}".format(features[i],features[j], val))
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        if(i<j and i != j):
            val = df[features[i]].cov(df[features[j]])
            print("Covariance for {} and {}: {}".format(features[i],features[j], val))
def correlation(x,y):
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    num = sum(sub_x[i] * sub_y[i] for i in range(len(sub_x)))
    std_x = sum(sub_x[i] **2.0 for i in range(len(sub_x)))
    std_y = sum(sub_y[i] **2.0 for i in range(len(sub_y)))
    denom = (std_x* std_y)** 0.5
    cor = num/denom
    return cor
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        if(i<j and i != j):
            val = correlation(df[features[i]], df[features[j]])
            print("Correlation for {} and {}: {}".format(features[i],features[j], val))
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        if(i<j and i != j):
            val = df[features[i]].corr(df[features[j]])
            print("Correlation for {} and {}: {}".format(features[i],features[j], val))
# Plotting the graph for above values
import matplotlib.pyplot as plt
import seaborn as sns
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        if(i<j and i != j):
            fig = plt.figure()
            fig.set_figheight(5)
            fig.set_figwidth(5)
            ax = sns.scatterplot(x = features[i], y = features[j], hue = "Species", data = df)
cor_matrix = df.corr(numeric_only = True)
round(cor_matrix, ndigits = 4)
sns.set(style = "whitegrid")
sns.heatmap(cor_matrix, annot = True, cmap = "grey", cbar = False, linecolor = "blue")
plt.show()
