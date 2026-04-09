import torch

# 1维
x = torch.tensor([10, 20, 30])
print("x =", x)
print("x.shape =", x.shape)
print("x[0] =", x[0])
print()

# 2维
y = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print("y =", y)
print("y.shape =", y.shape)
print("y[0] =", y[0])
print("y[1] =", y[1])
print("y[0, 1] =", y[0, 1])
print()

# 3维
z = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])
print("z =", z)
print("z.shape =", z.shape)
print("z[0] =", z[0])
print("z[1] =", z[1])
print("z[0, 1] =", z[0, 1])
print("z[0, 1, 2] =", z[0, 1, 2])
print()

# reshape
a = torch.tensor([1, 2, 3, 4, 5, 6])
print("a.shape =", a.shape)

b = a.reshape(2, 3)
print("b =", b)
print("b.shape =", b.shape)

c = a.reshape(3, 2)
print("c =", c)
print("c.shape =", c.shape)