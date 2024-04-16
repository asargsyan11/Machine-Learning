import pandas as pd

#Linear Regression with my own database

data = {
    'Price' : [13500, 13750, 13950, 14950, 13750, 12950, 16900, 18600, 21500, 12950],
    'Age' : [23, 23, 24, 26, 30, 32, 27, 30, 27, 23],
    'KM' : [46986, 72937, 41711, 48000, 38500, 61000, 94612, 75889, 19700, 71138],
    'Weight' : [1165.0 ,1165.0, 1165.0, 1165.0, 1170.0, 1170.0, 1245.0, 1245.0, 1185.0,1185.0]
}

df = pd.DataFrame(data)

X = df['Price']
y1 = df['Age']
y2 = df['KM']
y3 = df['Weight']

import matplotlib.pyplot as plt

plt.scatter(X,y1)
plt.xlabel("Price")
plt.ylabel("Age")


import matplotlib.pyplot as plt

plt.scatter(X,y2)
plt.xlabel("Price")
plt.ylabel("KM")


#process the data

from sklearn.model_selection import train_test_split

X_train, X_test, y1_train, y1_split = train_test_split(X, y1, test_size = 0.3, random_state = 10)

import numpy as np

X_train1 = np.array(X_train).reshape(-1,1)
X_test1 = np.array(X_test).reshape(-1,1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
c = lr.intercept_
m = lr.coef_
y_train_pred = lr.predict(X_train)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train_pred)
plt.plot(X_train,y_train_pred, color = 'red')

plt.xlabel("Price")
plt.ylabel("KM")

y_test_pred = lr.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X_test,y_test_pred)
plt.plot(X_test,y_test_pred, color = 'red')

plt.xlabel("Price")
plt.ylabel("KM")