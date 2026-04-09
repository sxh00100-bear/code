import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[2.0], [4.0], [6.0],[8.0]])

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(300):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"epoch={epoch}, loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")

print("_" * 50)
print("模型结构:")
print(model)
print()

print("weight:")
print(model.linear.weight)
print("weight.shape =", model.linear.weight.shape)
print()

print("bias:")
print(model.linear.bias)
print("bias.shape =", model.linear.bias.shape)
print()

print("model.parameters():")
for param in model.parameters():
    print(param)
    print("shape =", param.shape)
    print()