#Import Packages
import pandas as pd
import numpy as np

#Import Dataset
df = pd.read_csv('https://raw.githubusercontent.com/siiddd/LinearRegression/master/Cars.csv')

#Explore the Dataset
df.head(3)
df.describe()

#Check for Missing Values
df.isna().sum()

#Build a Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


#Visualizing the best fit line
import seaborn as sns
sns.regplot(x = df.speed, y = df.dist)

#Fit our data into the model
model.fit(pd.DataFrame(df.speed), pd.DataFrame(df.dist))

#Determine the Coefficient and Intercept of our Model
model.coef_
model.intercept_

#Predict Output for Custom Values of Input
model.predict([[25]])

