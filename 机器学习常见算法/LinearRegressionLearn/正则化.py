from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
np.random.seed(42)

m = 20
X = 3 * np.random.rand(m, 1)
y = 0.5 * X + np.random.randn(m, 1)/1.5 + 1

X_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alpha, **model_kargs):
    for alp, style in zip(alpha, ('b-', 'g--', 'r:')):
        model = model_class(alp, **model_kargs)
        if polynomial:
            model = Pipeline([('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
                              ("StandardScaler", StandardScaler()),
                              ("lin_reg", model)])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alp > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label="alpha={}".format(alp))
        plt.legend()
    plt.plot(X, y, 'b.', linewidth=3)

plt.figure(figsize=(14, 6))
plt.subplot(121)
# 岭回归 0,10,100表示正则化的粒度
plot_model(Ridge, polynomial=False, alpha=(0, 10, 100))
plt.subplot(122)
plot_model(Ridge, polynomial=True, alpha=(0, 10**-5, 1))
plt.show()

# lasso

plt.figure(figsize=(14, 6))
plt.subplot(121)
# Lasso 0,10,100表示正则化的粒度
plot_model(Lasso, polynomial=False, alpha=(0, 0.1, 1))
plt.subplot(122)
plot_model(Lasso, polynomial=True, alpha=(0, 10**-1, 1))
plt.show()