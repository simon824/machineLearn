import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([8, 7, 6, 5, 4, 3])
print(a)
print(a.shape)
c = np.reshape(a, (3, 2))
print(c)
print(c.shape)
d = np.reshape(b, (1, 1, 2, 3))
print(d, d.shape)
e = np.squeeze(d)
print(e, e.shape)
f = e.transpose((1, 0))
print(f)
index = np.argmax(f)
print(index, f[index])

arr = np.zeros((6, 6), np.uint8)
arr2 = np.linspace((6, 4, 2), 10, 10)
print(arr)
print(arr2)

if __name__ == '__main__':
    from varname import nameof
    green = 1000
    red = 2000
    blue = 3000
    lst = [red, green, blue]

