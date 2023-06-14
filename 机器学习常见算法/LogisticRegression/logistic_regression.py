import numpy as np
from scipy.optimize import minimize
from utils.features.prepare_for_training import prepare_for_training
from utils.hypothesis.sigmoid import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid=0, normalize_data=False):
        """
            1.对数据进行预处理操作
            2.先得到所有的特征个数
            3.初始化参数矩阵
        """
        (data_processed, feature_mean, feature_deviation) = prepare_for_training(data, polynomial_degree, sinusoid,
                                                                                 normalize_data)
        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.feature_mean = feature_mean
        self.feature_deviation = feature_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid = sinusoid
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = np.unique(labels).shape[0]
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        cost_history = []
        num_features = self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            current_labels = (self.labels == unique_label).astype(float)
            # 梯度下降
            (current_theta, cost_his) = LogisticRegression.gradient_descent(self.data, current_labels,
                                                                            current_initial_theta, max_iterations)
            self.theta[label_index] = current_theta.T
            cost_history.append(cost_his)
        return self.theta, cost_history

    @staticmethod
    def gradient_descent(data, current_labels, current_initial_theta, max_iterations):
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
            # 要优化的目标
            lambda current_theta: LogisticRegression.cost_function(data, current_labels, current_theta.reshape(num_features, 1)),
            # 初始化权重参数
            current_initial_theta,
            # 优化策略
            method='CG',
            # 梯度下降迭代计算公式
            jac=lambda current_theta: LogisticRegression.gradient_step(data, current_labels, current_theta.reshape(num_features, 1)),
            # 记录结果
            callback=lambda current_theta: cost_history.append(LogisticRegression.cost_function(data, current_labels, current_theta.reshape(num_features, 1))),
            # 迭代次数
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError("can not minimize cost function" + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, current_labels, current_initial_theta):
        num_examples = current_labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, current_initial_theta)
        label_diff = predictions - current_labels
        gradients = (1/num_examples) * np.dot(data.T, label_diff)
        return gradients.T.flatten()

    @staticmethod
    def cost_function(data, labels, theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    def predict(self, data):
        num_examples = data.shape[0]
        # 数据预处理
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid,
                                                                                 self.normalize_data)[0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)
        max_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_index == index] = label
        return class_prediction.reshape((num_examples, 1))
    


