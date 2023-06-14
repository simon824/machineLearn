import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from machineLearn.LinearRegressionLearn.linear_regression import LinearRegression

data = pd.read_csv("../data/non-linear-regression-x-y.csv")

x = data['x'].values.reshape(data.shape[0], 1)
y = data['y'].values.reshape(data.shape[0], 1)

num_iterations = 50000
learning_rate = 0.02
polynomial_degree = 15
sinusoid = 15
linear_regression = LinearRegression(x, y, polynomial_degree=polynomial_degree, sinusoid=sinusoid)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)
print("开始时的损失: ", cost_history[0])
print("训练后的损失： ", cost_history[-1])

theta_table = pd.DataFrame({"Model Parameters": theta.flatten()})

plt.plot(range(num_iterations), cost_history)
plt.xlabel("Iter")
plt.ylabel("cost")
plt.title("GD")
plt.show()

prediction_num = 1000
x_predictions = np.linspace(x.min(), x.max(), prediction_num).reshape(prediction_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x, y, label="Train data")
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.title("Happy")
plt.legend()
plt.show()