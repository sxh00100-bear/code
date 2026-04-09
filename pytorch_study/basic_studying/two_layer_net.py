import torch
import torch.nn as nn
import torch.optim as optim

# 数据：这里我们故意做一个非线性关系 y = x^2
x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y_true = torch.tensor([[4.0], [1.0], [0.0], [1.0], [4.0]])

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        #print("输入 x.shape =", x.shape)

        x = self.fc1(x)
        #print("经过 fc1 后 shape =", x.shape)

        x = self.relu(x)
        #print("经过 relu 后 shape =", x.shape)

        x = self.fc2(x)
        #print("经过 fc2 后 shape =", x.shape)
        return x


model = TwoLayerNet()

print("fc1.weight.shape =", model.fc1.weight.shape)
print("fc1.bias.shape =", model.fc1.bias.shape)
print("fc2.weight.shape =", model.fc2.weight.shape)
print("fc2.bias.shape =", model.fc2.bias.shape)
print()

y_pred = model(x)
print("\n最终输出 y_pred =")
print(y_pred)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

print("\n最终预测：")
print(model(x))

print("_" * 50)
print("fc1.weight:")
print(model.fc1.weight)
print("fc1.weight.shape =", model.fc1.weight.shape)
print()

print("fc1.bias:")
print(model.fc1.bias)
print("fc1.bias.shape =", model.fc1.bias.shape)
print()

print("fc2.weight:")
print(model.fc2.weight)
print("fc2.weight.shape =", model.fc2.weight.shape)
print()

print("fc2.bias:")
print(model.fc2.bias)
print("fc2.bias.shape =", model.fc2.bias.shape)
print()

print("所有参数：")
for name, param in model.named_parameters():
    print(name, param.shape)