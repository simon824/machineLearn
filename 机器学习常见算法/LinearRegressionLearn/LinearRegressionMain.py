import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 拼接
X_b = np.c_[np.ones((100, 1)), X]
# 求逆
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

X_new = np.array([[0], [2]])
print(X_new)
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("", lin_reg.coef_)
print("", lin_reg.intercept_)

# plt.plot(X, y, 'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([0, 2, 0, 15])
# plt.show()

# 批量梯度下降
eta = 0.1  # 学习率
n_iterations = 1000  # 迭代次数
m = 100  # 样本数量
theta = np.random.randn(2, 1)  #
for iteration in range(n_iterations):
    # 计算公式
    gradients = 2 / m * X_b.T.dot((X_b.dot(theta) - y))
    theta = theta - eta * gradients

res = X_new_b.dot(theta)
print(res)

# plt.plot(X_new, y_predict, 'r--')
# plt.plot(X, y, 'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([0, 2, 0, 15])
# plt.show()

theta_path_bgd = []


def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, 'b.')
    n_iterations = 1000
    theta = np.random.randn(2, 1)  #
    for iteration in range(n_iterations):
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new, y_predict, 'g--')
        # 计算公式
        gradients = 2 / m * X_b.T.dot((X_b.dot(theta) - y))
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path_bgd.append(theta)
    plt.xlabel("X_1")
    plt.axis([0, 2, 0, 15])
    plt.title("eta = {}".format(eta))


theta = np.random.randn(2, 1)
plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)
plt.show()
print("theta_path_bgd", theta_path_bgd)

theta_path_sgd = []
m = len(X_b)
n_epoch = 50
t0 = 5
t1 = 50


def learning_schedule(t):
    return t0 / (t1 + t)

# 随机梯度下降
theta = np.random.randn(2, 1)

for epoch in range(n_epoch):
    for i in range(m):
        if epoch < 10 and i < 10:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, 'r--')
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(n_epoch*m+i)
        theta = theta -eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()


# 小批量梯度下降
theta_path_mgd = []
n_epoch = 50
minbatch = 16  # 2的次幂
theta = np.random.randn(2, 1)
np.random.seed(0)  # 使随机结果相同
t = 0
for epoch in range(n_epoch):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minbatch):
        t += 1
        xi = X_b_shuffled[i:i + minbatch]
        yi = y_shuffled[i:i+minbatch]
        gradients = 2/minbatch * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# 3种策略的对比实验
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

print("theta_path_bgd", theta_path_bgd)
plt.figure(figsize=(8, 4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], 'r-s', linewidth=1, label='SGD')
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'g-+', linewidth=2, label='MGD')
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], 'b-o', linewidth=3, label='BGD')
plt.legend(loc='upper left')
plt.axis([3.5, 4.5, 2.0, 4.0])
plt.show()

