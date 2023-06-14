import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

x = np.array([1, 2, 0.5, 2.5, 2.6, 3.1], np.float32).reshape((-1, 1))
y = np.array([3.7, 4.6, 1.65, 5.68, 5.98, 6.95], np.float32).reshape((-1, 1))


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
# 模型
model = LinearRegressionModel(input_dim, output_dim)
# 损失函数
criterion = nn.MSELoss()

learning_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(100):
    epoch += 1
    inputs = torch.from_numpy(x).requires_grad_()
    labels = torch.from_numpy(y)
    # 梯度清零
    optimizer.zero_grad()
    # 调用forward函数
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向梯度
    loss.backward()
    # 通过优化器，更新参数
    optimizer.step()
    print("epoch {}, loss {}".format(epoch, loss.item()))

predicted_y = model(torch.from_numpy(x).requires_grad_()).data.numpy()
print("标签Y: ", y)
print("预测Y： ", predicted_y)

plt.clf()
predicted = model(torch.from_numpy(x).requires_grad_()).data.numpy()
# 绘制真实值
plt.plot(x, y, "go", label="True data", alpha=0.5)
# 绘制预测值
plt.plot(x, predicted_y, "--", label="Predictions", alpha=0.5)

plt.legend(loc="best")
plt.show()