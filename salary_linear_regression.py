from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")

x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_test)
ln=LinearRegression()
ln.fit(x_train,y_train)
predict=ln.predict(x_test)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,ln.predict(x_train),color="blue")

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,ln.predict(x_train),color="blue")
plt.show()