import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

dataframe = pd.read_csv('Position_Salaries.csv')
#print(dataframe.head())
X = dataframe.iloc[:,1:2].values
Y = dataframe.iloc[:,2].values

reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X,Y)

X_pre = 6.5

pred = reg.predict(X_pre)

X_adj = np.arange(min(X),max(X),0.001)
X_adj = X_adj.reshape((len(X_adj)),1)

plt.scatter(X,Y,s = 50,color = 'red')
plt.plot(X_adj,reg.predict(X_adj),color = 'blue')
plt.scatter(X_pre,pred,color = 'yellow')
plt.show()



