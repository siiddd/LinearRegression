#Import Packages
import pandas as pd
import numpy as np

#Import the Dataset
df = pd.read_csv('https://raw.githubusercontent.com/siiddd/LinearRegression/master/Crime.csv')

#Check for Missing Values
df.isna().sum()

import scipy.stats
scipy.stats.mode(df.Ed)[0][0]

#Replacing the Missing Values in our dataset
df.fillna(scipy.stats.mode(df.Ed)[0][0], inplace = True)

#Detect Outliers
import seaborn as sns
sns.boxplot(data = df.N)
from scipy.stats import iqr

#Treat Outliers
for j in range(0,14):
    for i in range(0,47):
        if(df.iloc[i,j] > np.percentile(df.iloc[:, j], 75) + 1.5*iqr(df.iloc[:, j])):
            df.iloc[i,j] = np.percentile(df.iloc[:,j], 90)
            
for j in range(0,14):
    for i in range(0,47):
        if(df.iloc[i,j] < np.percentile(df.iloc[:, j], 25) - 1.5*iqr(df.iloc[:, j])):
            df.iloc[i,j] = np.percentile(df.iloc[:,j], 10)
          
#Detect and Treat Multicollinearity            
df_corr = abs(df.corr().round(2))
df.columns.values

features = ['R', 'Age', 'Ed', 'Ex0', 'M', 'N', 'U2'] #Ex1, X, W, U1, S, NW, LF
features_withoutR = ['Age', 'Ed', 'Ex0', 'M', 'N', 'U2'] #Ex1, X, W, U1, S, NW, LF

df_corr_updated = abs(df.loc[:, features].corr().round(2))
df_corr_withoutR = abs(df.loc[:, features_withoutR].corr().round(2))

#Updated Dataset (Clean)
df_clean = df.loc[:, features]

#Creation of Test and Train Dataset
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(df_clean.iloc[:, [1,2,3,4,5,6]], df_clean.iloc[:, 0], train_size = 0.90)      
          
