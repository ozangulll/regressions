# in simple linear regression y=b0+b1*x
#maas=b0+b1*deneyim
# in multiple linear regression
#y=b0+b1*x1+b2*x2

#mass yani y değeri aslında burda dependent variable olarak geçmektedir.
#deneyim ve yas değiskenlerimzi de independent variableo larak geçmektedir.
#b0,b1,b2
# AMACIMIZ MEAN SQUARE ERROR'umuzu minimuma düşürmek

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df=pd.read_csv('multiple_linear_regression_dataset.csv',sep=";")
x = df.iloc[:, [0, 2]].values  # Adjusted to select columns as 2D
y = df['maas'].values.reshape(-1, 1)

multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0",multiple_linear_regression.intercept_)
print("b1,b2",multiple_linear_regression.coef_)
print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))
