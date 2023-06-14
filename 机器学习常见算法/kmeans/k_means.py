import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.clusters = num_clusters

    def train(self, max_iterations):
        # 随机选择K个中心点
        centroids = KMeans.centroids_init(self.data, self.clusters)
        # 开始训练
        num_examples = self.data.shape[0]
        closest_centroids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 得到当前每个样本点到k个中心点的距离，找到最近的中心点
            closest_centroids = KMeans.centroids_find_closest(self.data, centroids)
            # 中心点位置更新
            centroids = KMeans.centroids_compute(self.data, closest_centroids, self.clusters)
        return centroids, closest_centroids

    @staticmethod
    def centroids_init(data, clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:clusters], :]
        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))
            for centroids_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroids_index, :]
                distance[centroids_index] = np.sum(distance_diff**2)
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids, clusters):
        num_features = data.shape[1]
        centroids = np.zeros((clusters, num_features))
        for centroid in range(clusters):
            closest_ids = closest_centroids == centroid
            centroids[centroid] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centroids





