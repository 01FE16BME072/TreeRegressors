import numpy as np
import pandas as pf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataframe = pf.read_csv('Position_Salaries.csv')

X = dataframe.iloc[:,1:2].values
Y = dataframe.iloc[:,2].values

reg = RandomForestRegressor(n_estimators = 300,random_state = 0)

reg.fit(X,Y)

X_best = np.arange(min(X),max(X),0.0001)
X_best = X_best.reshape(len(X_best),1)

plt.scatter(X,Y,s = 50,color = 'red')
plt.plot(X_best,reg.predict(X_best),color = 'blue')


X_pred = 6.5
y_pred = reg.predict(X_pred)
print(y_pred)
plt.scatter(X_pred,y_pred,s = 100,color = 'green')
plt.show()