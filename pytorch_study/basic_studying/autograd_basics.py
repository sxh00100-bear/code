import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

print("x =", x)
print("y =", y)

y.backward()

print("x.grad =", x.grad)
print("_"*50)

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1

print("x =", x)
print("y =", y)

y.backward()

print("x.grad =", x.grad)

print("_"*50)
x = torch.tensor(2.0, requires_grad=True)

a = x * 3
b = a ** 2

print("x =", x)
print("a =", a)
print("b =", b)

b.backward()

print("x.grad =", x.grad)
print("_"*50) 

x = torch.tensor(2.0, requires_grad=True)
y1 = x ** 2
y1.backward()
print("x = ", x)
print("y1 =", y1)
print("第一次 backward 后 x.grad =", x.grad)

y2 = x ** 2
y2.backward()
print("第二次 backward 后 x.grad =", x.grad)