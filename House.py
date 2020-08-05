#Import the packages
import pandas as pd
import numpy as np


#Import the dataset
df = pd.read_csv('https://raw.githubusercontent.com/siiddd/LinearRegression/master/House.csv')

#Explore the Dataset
df.describe()

#Check for Missing Values
df.isna().sum()

#Fill Missing Values
np.mean(df.bedrooms)
np.median(df.bedrooms)

import scipy.stats
scipy.stats.mode(df.bedrooms)
df['bedrooms'].fillna(scipy.stats.mode(df.bedrooms)[0][0], inplace = True)

#Draw the Best Fit Line
import seaborn as sns
sns.regplot(x = df.age, y = df.price)
sns.regplot(x = df.bedrooms, y = df.price)
sns.regplot(x = df.area, y = df.price)


#Build a Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


#Fit the Data into the Model
model.fit(df.iloc[:, [0,1,2]], df.iloc[:, -1])

#Determine the Coefficient and the Intercept
model.coef_
model.intercept_

#Predict for Custom Input Data
model.predict([[3000,3,40]])

