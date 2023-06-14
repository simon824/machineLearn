import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets


iris = datasets.load_iris()

# 选取2 3 特征
X, y = iris['data'][:, (2, 3)], iris['target']

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel='linear', C=float(2 ** 1000))
svm_clf.fit(X, y)

x0 = np.linspace(0, 5.5, 200)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

def plot_svc_desicion_boundary(svm_clf, xmin, xmax, sv=True):
    w = svm_clf.coef_[0]  # 权重参数
    b = svm_clf.intercept_[0]  # 偏置参数
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = - w[0] / w[1] * x0 - b / w[1]
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    if sv:
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')

    plt.plot(x0, decision_boundary, 'k-', linewidth=2)
    plt.plot(x0, gutter_up, 'k--', linewidth=2)
    plt.plot(x0, gutter_down, 'k--', linewidth=2)

plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ys')
plt.plot(x0, pred_1, 'g--', linewidth=2)
plt.plot(x0, pred_2, 'm-', linewidth=2)
plt.plot(x0, pred_3, 'r-', linewidth=2)
plt.axis([0, 5.5, 0, 2])

plt.subplot(122)
plot_svc_desicion_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ys')
plt.axis([0, 5.5, 0, 2])
plt.show()


# 可以使用超参数控制软间隔程度
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

X, y = iris['data'][:, (2, 3)], (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline((
    ('std', StandardScaler()),
    ("linear_svc", LinearSVC(C=1))
))

svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])

scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, random_state=42)
svm_clf2 = LinearSVC(C=100, random_state=42)

scaled_svm_clf1 = Pipeline((
    ('std', StandardScaler()),
    ("linear_svc", svm_clf1)
))

scaled_svm_clf2 = Pipeline((
    ('std', StandardScaler()),
    ("linear_svc", svm_clf2)
))

def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1) ** 2)





