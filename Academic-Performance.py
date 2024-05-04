import pandas as pd
test = pd.read_csv('/home/meghan/Downloads/AP.csv')
test
missing_values = test.isnull().sum()
print(missing_values)
miss_val_per = (100 * missing_values)/ len(missing_values)
print(miss_val_per)
import missingno as msno
msno.bar(test)
import numpy as np
test['COURSE 1 MARKS'] = test['COURSE 1 MARKS'].replace(np.NaN,test['COURSE 1 MARKS'].mean())
test['COURSE 2 MARKS'] = test['COURSE 2 MARKS'].fillna(0)
test['COURSE 4 MARKS'] = test['COURSE 4 MARKS'].fillna(0)
missing_values = test.isnull().sum()
print(missing_values)
Y = pd.DataFrame(test['ACADEMIC_PROGRAM'])
Y
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "most_frequent")
imputer.fit_transform(Y)
imputer = SimpleImputer(strategy = "constant", fill_value = "missing")
imputer.fit_transform(Y)
mv = test.isnull().sum()
print(mv)
import seaborn as sns
sns.boxplot(data = test, x = test['COURSE 1 MARKS'])
import matplotlib.pyplot as plt
plt.boxplot(test['COURSE 1 MARKS'], vert = False)
plt.xlabel("test['COURSE 1 MARKS']")
plt.title('Detecting outliers')
import numpy as np
outliers = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        zscore = (i-mean)/std
        if(np.abs(zscore) > thres):
            outliers.append(i)
    return outliers
marks_outliers = detect_outliers_zscore(test['COURSE 1 MARKS'])
print(f"Marks Detected by zscore Method is {marks_outliers}")
outliers = []
def detect_outliers_iqr(data):
    data = sorted(data)
    Q1 = np.percentile(data,25)
    Q3 = np.percentile(data,75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5*IQR)
    upr_bound = Q3 + (1.5*IQR)
    for i in data:
        if((i<lower_bound) or (i>upr_bound)):
            outliers.append(i)
    return outliers
marks_outliers = detect_outliers_iqr(test['COURSE 1 MARKS'])
print(f"Marks detected by IQR are {marks_outliers}")
Q1 = test['COURSE 1 MARKS'].quantile(0.25)
Q3 = test['COURSE 1 MARKS'].quantile(0.75)
print(Q1)
print(Q3)
IQR = Q3 - Q1
print(IQR)
lower_whisker = Q1- (1.5*IQR)
upr_whisker = Q3 + (1.5*IQR)
print(lower_whisker, upr_whisker)
test = test[test['COURSE 1 MARKS']< upr_whisker]
test
import numpy as np
performance_catogorical = test.select_dtypes(exclude = (np.number))
performance_catogorical
performance_catogorical.ACADEMIC_PROGRAM.value_counts()
performance_catogorical.OVEARLL_GRADE.value_counts()
performance_catogorical.PLACEMENT.replace({"Yes":1, "No":-1}, inplace = True)
performance_catogorical = performance_catogorical.drop("STUDENT_ID", axis = 1)
performance_catogorical
data_column_category = performance_catogorical.select_dtypes(exclude=(np.number)).columns
data_column_category
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in data_column_category:
    performance_catogorical[i] = label_encoder.fit_transform(performance_catogorical[i])
print("Label Encoded Data:")
performance_catogorical.head()
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(performance_catogorical[data_column_category])
onehot_encoded_frame = pd.DataFrame(onehot_encoded, columns = onehot_encoder.get_feature
onehot_encoded_frame.head()
