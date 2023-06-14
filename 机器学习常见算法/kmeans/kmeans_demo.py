import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from k_means import KMeans

data = pd.read_csv("../data/Iris.csv")
iris_types = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(data.columns)
print(data['Species'].unique())

x_axis = 'PetalLengthCm'
y_axis = 'PetalWidthCm'

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species'] == iris_type], data[y_axis][data['Species'] == iris_type], label=iris_type)
plt.title('label know')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('label unknow')
plt.scatter(data[x_axis][:], data[y_axis][:])
# plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))

# 指定训练所需的参数
num_clusters = 3
max_iterations = 50

k_means = KMeans(x_train, num_clusters)
centroids, closest_centroids = k_means.train(max_iterations)

# 对比结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species'] == iris_type], data[y_axis][data['Species'] == iris_type], label=iris_type)
plt.title('label know')
plt.legend()

plt.subplot(1, 2, 2)

for index, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids == index).flatten()
    plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=index)

for index, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')
plt.title('label kmeans')
plt.legend()
plt.show()
