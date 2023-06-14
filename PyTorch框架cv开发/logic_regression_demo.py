import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 20, np.float32)
_b = 1 / (1 + np.exp(-x))
y = np.random.normal(_b, 0.005)

x = np.float32(x.reshape((-1, 1)))
y = np.float32(y.reshape((-1, 1)))

class LogicRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogicRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

input_dim = 1
output_dim = 1
# sigmoid+BCE”对应的是torch.nn.BCEWithLogitsLoss，而“softmax+CE”对应的是torch.nn.CrossEntropyLoss
# 模型
model = LogicRegressionModel(input_dim, output_dim)
# 损失函数
criterion = torch.nn.BCELoss()
# 学习率
learning_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# sigmoid -> 1 / (1 + e^(-x))
for epoch in range(100):
    epoch += 1
    inputs = torch.from_numpy(x).requires_grad_()
    labels = torch.from_numpy(y)
    # 梯度清零
    optimizer.zero_grad()
    # 模型预测, 调用forward函数
    predict = model(inputs)
    # 计算损失
    loss = criterion(predict, labels)
    # 反向梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    print("epoch {}, loss {}".format(epoch, loss.item()))

# 进行预测
predicted = model(torch.from_numpy(x).requires_grad_()).data.numpy()
print("标签Y: ", y)
print("预测Y: ", predicted)

plt.clf()
predicted_ = model(torch.from_numpy(x).requires_grad_()).data.numpy()
# print(list(zip(predicted, predicted_)))
plt.plot(x, y, "go", label="True data", alpha=0.5)
plt.plot(x, predicted, "--", label="Predict1", alpha=0.5)
plt.plot(x, predicted_, "-", label="Predict2", alpha=0.5)
plt.legend(loc="best")
plt.show()