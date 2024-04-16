'''
My first time trying to work with KNN so I'm not sure how well this is written
'''

from sklearn.datasets import load_diabetes
import pandas as pd

diabetes = load_diabetes()
data = pd.DataFrame(data = diabetes.data,columns = diabetes.feature_names)
data['target'] = diabetes.target
data

X = data['bmi'].values.reshape(-1,1)
y = data['target']

import matplotlib.pyplot as plt
plt.scatter(X,y, color = "#607D3B")
plt.xlabel("bmi")
plt.ylabel("target")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 100)

from sklearn.neighbors import KNeighborsRegressor

KNN = KNeighborsRegressor(n_neighbors= 3)
KNN.fit(X_train,y_train)

y_KNN_train_predict = KNN.predict(X_train)
y_KNN_test_predict = KNN.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

KNN_train_MSE = mean_squared_error(y_train, y_KNN_train_predict)
KNN_test_MSE = mean_squared_error(y_test,y_KNN_test_predict)
KNN_train_r2 = r2_score(y_train, y_KNN_train_predict)
KNN_test_r2 = r2_score(y_test,y_KNN_test_predict)

print("The training MSE is: ",KNN_train_MSE)
print("The training R2 is: ",KNN_train_r2)
print("The testing MSE is: ", KNN_test_MSE)
print("The testing R2 is: ",KNN_test_r2)

KNN_results = pd.DataFrame(['KNN',KNN_train_MSE, KNN_train_r2, KNN_test_MSE, KNN_test_r2]).transpose()
KNN_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
KNN_results