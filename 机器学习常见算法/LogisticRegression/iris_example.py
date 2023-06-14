import numpy as np

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(iris.keys())
# print(iris['data'])
X = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)

log_res = LogisticRegression(C=1000)
log_res.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_res.predict_proba(X_new)

plt.figure(figsize=(12, 5))
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]
plt.plot([decision_boundary, decision_boundary], [-1, 2], 'k:', linewidth=1)
plt.plot(X_new, y_proba[:, 1], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Not Iris-Virginica')
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.text(decision_boundary + 0.02, 0.15, 'decision boundary', fontsize=14, color='k', ha='center')
plt.xlabel("Peta width(cm)", fontsize=14)
plt.ylabel("y_proba", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.legend(loc='center left', fontsize=14)
plt.show()

X = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.int)
log_res = LogisticRegression()
log_res.fit(X, y)

# 构建坐标数据
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1), np.linspace(0.8, 2.7, 200).reshape(-1, 1))

X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_res.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 0, 0], X[y == 0, 1], 'bs')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'g^')

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()





