import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
softmax_reg.fit(X, y)

# 构建坐标数据
x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1), np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)
zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], 'g^', label='Iris-Virginica')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'bs', label='Iris-Versicolor')
plt.plot(X[y == 0, 0], X[y == 0, 1], 'yo', label='Iris-Setosa')

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

plt.contour(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.xlabel("Peta length(cm)", fontsize=14)
plt.ylabel("Peta width", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.legend(loc='center left', fontsize=14)
plt.show()
