#Import Packages
import pandas as pd
import numpy as np

#Import Dataset
df = pd.read_csv('https://raw.githubusercontent.com/siiddd/LinearRegression/master/Trees.csv')

#Explore the Dataset
df.describe()
df.head(3)
df.shape

#Draw the Best Fit Line
import seaborn as sns
sns.regplot(x = df.Height, y = df.Volume)
sns.regplot(x = df.Girth, y = df.Volume)

#Build a Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Fit our Data in to the Model
model.fit(df.iloc[:,[0,1]],df.iloc[:,-1])

#Get the Output For Custom Input Values
model.predict([[11,66]])
