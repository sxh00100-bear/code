import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print("x =", x)
print("x.shape =", x.shape)

y = x.reshape(2, 3)
print("y =", y)
print("y.shape =", y.shape)
print("_"*50)
z = y.unsqueeze(0)
print("z =", z)
print("z.shape =", z.shape)
zz = z.unsqueeze(0)
print("zz =", zz)
print("zz.shape =", zz.shape)
zzz = zz.unsqueeze(0)
print("zzz =", zzz)
print("zzz.shape =", zzz.shape)

x = torch.randn(1, 2, 1, 4)
print("x.shape =", x.shape)

print("x.squeeze().shape =", x.squeeze().shape)
print("x.squeeze(0).shape =", x.squeeze(0).shape)
print("x.squeeze(1).shape =", x.squeeze(1).shape)
print("x.squeeze(2).shape =", x.squeeze(2).shape)

y = torch.tensor([1, 2, 3, 4])
print("y.shape =", y.shape)
print("y.unsqueeze(0).shape =", y.unsqueeze(0).shape)
print("y.unsqueeze(1).shape =", y.unsqueeze(1).shape)
print("_"*50)


w = z.squeeze(0)
print("w =", w)
print("w.shape =", w.shape)
print("_"*50)
a = torch.randn(2, 3)
b = torch.randn(2, 3)

print("a =", a)
print("b =", b)
print("a + b =", a + b)
print("a * b =", a * b)
print("a.mean() =", a.mean())
print("a.sum() =", a.sum())