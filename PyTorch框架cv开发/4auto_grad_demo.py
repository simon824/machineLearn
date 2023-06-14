import torch

# https://deeplizard.com/learn/video/Csa5R12jYRg
# requires_grad 是否自动求导
x = torch.randn(1, 5, requires_grad=True)
y = torch.randn(5, 3, requires_grad=True)
z = torch.randn(3, 1, requires_grad=True)
print("x: \n", x, "\ny: \n", y, "\nz: \n", z)
# 矩阵相乘
xy = torch.matmul(x, y)
print("xy: \n", xy)
xyz = torch.matmul(xy, z)
print("xyz: ", xyz)
# 实现反向求导操作
xyz.backward()
# z的梯度 z.grad == xy == xyz对z求导
print(x.grad, y.grad, z.grad)