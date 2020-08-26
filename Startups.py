#Import Packages
import pandas as pd
import numpy as np

#Import the Dataset
df = pd.read_csv('https://raw.githubusercontent.com/siiddd/LinearRegression/master/Startups.csv')

#Explore the Dataset
df.describe()
df.head(3)

#Check for Missing Values
df.isna().sum()

#Check for Outliers
import seaborn as sns
sns.boxplot(data = df)

df.iloc[:,0][df.iloc[:,0] > df.iloc[:,0].quantile(0.95)]
df.iloc[:,1][df.iloc[:,1] > df.iloc[:,1].quantile(0.95)]
df.iloc[:,2][df.iloc[:,2] > df.iloc[:,2].quantile(0.95)]

#Creating Dummy Variables
df_dummy = pd.get_dummies(df.State)

#Dataset with Dummy Variables
df = pd.concat([df, df_dummy], axis = 1)
df.drop('Marketing Spend', axis = 1, inplace = True) #Drop State and California prior to Marketing Spend

#Check for MultiCollinearity
df_corr = df.iloc[:,[0,1,2,3]].corr()


#Create a Train and (Test) Validation Dataset
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(df.iloc[:, [0,1,-2,-1]], df.iloc[:, 2], train_size = 0.90)


#Build a Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#Train the model
lr.fit(x_train, y_train)

#Test the model
lr.score(x_test, y_test)

#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score 
cross_val_score(lr, df.iloc[:, [0,1,-2,-1]], df.iloc[:, 2], cv=3)
np.mean(cross_val_score(lr, df.iloc[:, [0,1,-2,-1]], df.iloc[:, 2], cv=3))

#Predict for Custom Values
lr.predict([[144372, 118672, 0,1]])
