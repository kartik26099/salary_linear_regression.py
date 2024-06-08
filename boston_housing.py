import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\boston housing.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)
print(y_predict)
print(y_test)