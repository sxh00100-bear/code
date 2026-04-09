import torch
import torch.nn as nn

torch.manual_seed(42)

class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DropoutNet()

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

print("输入 x =", x)
print()

# 训练模式
model.train()
print("=== train mode ===")
for i in range(3):
    y = model(x)
    print(f"第{i+1}次输出:", y)

print()

# 评估模式
model.eval()
print("=== eval mode ===")
for i in range(3):
    y = model(x)
    print(f"第{i+1}次输出:", y)