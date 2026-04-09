import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor([self.x_data[idx]], dtype=torch.float32)
        y = torch.tensor([self.y_data[idx]], dtype=torch.float32)
        return x, y

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 原始数据
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

# 数据集和加载器
dataset = MyDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 模型、损失、优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练
for epoch in range(300):
    epoch_loss = 0.0

    for x, y in dataloader:
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 20 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"epoch={epoch}, epoch_loss={epoch_loss:.4f}, w={w:.4f}, b={b:.4f}")

print("\n测试预测:")
test_x = torch.tensor([[6.0], [7.0]], dtype=torch.float32)
print("test_x =", test_x)
print("pred =", model(test_x))