import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device =", device)

# 超参数
batch_size = 64
learning_rate = 0.001
epochs = 5

# 数据预处理
transform = transforms.ToTensor()

# 数据集
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = x.view(x.size(0), -1)   # -> [batch, 784]
        x = self.fc1(x)             # -> [batch, 128]
        x = self.relu(x)            # -> [batch, 128]
        x = self.fc2(x)             # -> [batch, 10]
        return x

model = MLP().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练函数
def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"epoch={epoch}, batch={batch_idx}, loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"epoch={epoch} 平均loss={avg_loss:.4f}")

# 测试函数
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            total += y.size(0)
            correct += (preds == y).sum().item()

    acc = 100.0 * correct / total
    print(f"测试集准确率: {acc:.2f}%")

if __name__ == "__main__":
    train()
    evaluate()