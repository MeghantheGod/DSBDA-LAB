import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/AgeIncome.csv")
df
# Without using Library Functions
df.isnull().sum()
def mean(x):
    y = sum(x)/len(x)
    return y
mean(df['Income'])
def median(x):
    n = len(x)
    if n % 2:
        scores_mean = sorted(x)[round(0.5*(n-1))]
    else:
        x_ord, index = sorted(x), round(0.5*n)
        scores_mean = 0.5*(x_ord[index-1] + x_ord[index])
        return scores_mean
median(df['Income'])
minimum_income = min(df['Income'])
for i in range(len(df)):
    if df['Income'][i] == minimum_income:
        print(f"{df['Age'][i]}:{df['Income'][i]}")
maximum_income = max(df['Income'])
for i in range(len(df)):
    if df['Income'][i] == maximum_income:
        print(f"{df['Age'][i]}:{df['Income'][i]}")
import math
def standard_deviation(x):
    n = len(x)
    score_mean = sum(x)/n
    score_var = sum((item - score_mean)**2 for item in x)/ (n-1)
    score_std = math.sqrt(score_var)
    return round(score_std, ndigits=2)
standard_deviation(df['Income'])
def variance(x):
    n = len(x)
    score_mean = sum(x)/n
    score_var = sum((item - score_mean)**2 for item in x)/ (n-1)
    return round(score_var, ndigits=2)
variance(df['Income'])
# With using library functions
import numpy as np
import statistics
mean = np.mean(df['Income'])
mean
median = np.median(df['Income'])
median
standard_deviation = statistics.stdev(df['Income'])
print(round(standard_deviation, ndigits = 2))
variance = statistics.variance(df['Income'])
print(f"Variance is {round(variance, ndigits = 2)}")
groups = df.groupby('Age')
df.groupby('Age')['Income'].mean()
df.groupby('Age')['Income'].median()
df.groupby('Age')['Income'].std()
df.groupby('Age')['Income'].var()
