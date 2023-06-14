import numpy as np
from machineLearn.utils.features.prepare_for_training import prepare_for_training
from machineLearn.utils.hypothesis import sigmoid, sigmoid_gradient


class multilayerPerception:
    def __init__(self, data, labels,layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)
        self.data = data_processed
        self.labels = labels
        self.layers = layers  # 784  100 10
        self.normalize_data = normalize_data
        self.thetas = multilayerPerception.thetas_init(layers)

    def train(self, max_iterations=1000, alpha=0.1):
        unrolled_theta = multilayerPerception.theta_unroll(self.thetas)
        optimized_theata, cost_hist = multilayerPerception.gradient_descent(self.data, self.labels, self.layers, unrolled_theta, max_iterations, alpha)
        self.thetas = multilayerPerception.theta_roll(optimized_theata, self.layers)
        return self.thetas, cost_hist

    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        # 权重参数   784 * 100 、 100 * 10  两组
        for layer_index in range(num_layers-1):
            """
            会执行两次，得到两组参数矩阵，25 * 785, 10 * 26
            """
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            # 需要考虑偏置项，记住偏置的个数跟输出的结果是一致的
            thetas[layer_index] = np.random.rand(out_count, in_count+1)*0.05  # 随机进行初始化操作，尽量小一点

    @staticmethod
    def theta_unroll(thetas):
        """
        将所有层拉成一行  785 * 100 * 10 * 3
        """
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            np.hstack(unrolled_theta, thetas[theta_layer_index].flatten())
        return unrolled_theta

    @staticmethod
    def gradient_descent(data, labels, layers, unrolled_theta, max_iterations, alpha):
        optimized_theata = unrolled_theta
        cost_hist = []
        theta_rolled = multilayerPerception.theta_roll(unrolled_theta, layers)
        for _ in range(max_iterations):
            cost = multilayerPerception.cost_function(data, labels, theta_rolled, layers)
            cost_hist.append(cost)
            theta_gradient = multilayerPerception.gradient_setp(data, labels, optimized_theata, layers)
            optimized_theata = optimized_theata - alpha * theta_gradient
        return optimized_theata, cost_hist

    @staticmethod
    def gradient_setp(data, labels, optimized_theata, layers):
        theta = multilayerPerception.theta_roll(optimized_theata, layers)
        thetas_rolled_gradients = multilayerPerception.back_propagation(data, labels, theta, layers)
        thetas_unrolled_gradients = multilayerPerception.theta_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    @staticmethod
    def back_propagation(data, labels, theta, layers):
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        num_label_types = layers[-1]
        deltas = {}
        for layer_index in range(num_layers-1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            deltas[layer_index] = np.zeros((out_count, in_count+1))  # 100 * 785 10 * 101
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index, :].reshape((num_features, 1))  # 785 * 1
            layers_activations[0] = layers_activation
            # 逐层计算
            for layer_index in range(num_layers-1):
                layer_theta = theta[layer_index]  # 得到当前权重参数值
                layer_input = np.dot(layer_theta, layers_activation)  # 25 * 1   10 * 1
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                layers_inputs[layer_index+1] = layer_input  # 后一层计算结果
                layers_activations[layer_index+1] = layers_activation  # 后一层经过激活函数后的结果
            output_layer_activation = layers_activations[1:, :]
            delta = {}
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1
            # 计算输出层和真实值之间的差异
            delta[num_layers - 1] = output_layer_activation - bitwise_label
            # 遍历循环
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = theta[layer_index]
                next_delta = delta[layer_index+1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack(np.array((1)), layer_input)
                # 按照公式计算
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                # 过滤掉偏置项
                delta[layer_index] = delta[layer_index][1:, :]
            for layer_index in range(num_layers-1):
                layer_delta = np.dot(delta[layer_index+1], layers_activation[layer_index].T)
                delta[layer_index] = delta[layer_index] + layer_delta
        for layer_index in range(num_layers-1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)
        return deltas

    @staticmethod
    def theta_roll(unrolled_thetas, layers):
        num_layers = len(layers)
        unrolled_shift = 0
        thetas = {}
        for layer_index in range(num_layers-1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_height * thetas_width
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))
            unrolled_shift = unrolled_shift + thetas_volume
        return thetas

    @staticmethod
    def cost_function(data, labels, theta_rolled, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]

        # 前向传播
        predictions = multilayerPerception.feedforward_propagation(data, theta_rolled, layers)
        bitwise_labels = np.zeros((num_examples, num_labels))
        # 制作标签，每个标签都是one-hot编码
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
        bit_set_const = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_const = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = (-1 / num_examples) * (bit_set_const + bit_not_set_const)
        return cost

    @staticmethod
    def feedforward_propagation(data, theta_rolled, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data

        # 逐层计算
        for layer_index in range(num_layers-1):
            theta = theta_rolled[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            # 要考虑偏置项
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation
        # 返回输出层结果, 过滤偏置项
        return in_layer_activation[:, 1:]



