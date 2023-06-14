import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + np.random.randn(m, 1)
plt.plot(X, y, 'b.')
plt.xlabel('X_1')
plt.ylabel('y')
plt.axis([-3, 3, -5, 10])
plt.show()

poly_feature = PolynomialFeatures(degree=2, include_bias=False)
# 数据会多平方的一列
X_poly = poly_feature.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.coef_)  # 权重参数
print(lin_reg.intercept_)  # 偏置项

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_feature.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, 'r--', label='prediction')
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()

for style, width, degree in (('g-', 1, 100), ('b--', 1, 2), ('r-+', 1, 1)):
    poly_feature = PolynomialFeatures(degree=degree, include_bias=False)
    std = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_reg = Pipeline([('poly_feature', poly_feature),
                               ("StandardScaler", std),
                               ("lin_reg", lin_reg)
                               ])
    polynomial_reg.fit(X, y)
    y_new_2 = polynomial_reg.predict(X_new)
    plt.plot(X_new, y_new_2, label=str(degree), linewidth=width)

plt.plot(X, y, 'b.')
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()




