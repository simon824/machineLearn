import torch

# 生成空的tensor
x = torch.empty(2, 2)
# 生成满足正态分布(0-1)的随机数
x = torch.randn(2, 2)

# 生成0-1之间的随机数
x = torch.rand(2, 2)

# 初始化为0
x = torch.zeros(2, 2)

# 自定义
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

y = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])

# tensor相加
z = x.add(y)

# 维度变换
x = x.view(-1, 3)

# tensor数据转为numpy
nx = x.numpy()

# numpy数据转为tensor
x = torch.from_numpy(nx)

# 查看是否有GPU
if torch.cuda.is_available():
    print("GPU Detected!")
    # 使用GPU计算
    result = x.view(-1).cuda() + y.cuda()
    gpu_num = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_num)
    print("gpu_num:", gpu_num, "gpu_name:", gpu_name)
    print(result)
print(x, x.size())


