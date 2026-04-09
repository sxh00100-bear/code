import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device =", device)

batch_size = 64
learning_rate = 0.001
epochs = 5

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
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

# 把原来的训练集拆成 train 和 val
train_size = 55000
val_size = 5000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)   # [batch, 8, 28, 28]
        x = self.relu1(x)
        x = self.pool1(x)   # [batch, 8, 14, 14]

        x = self.conv2(x)   # [batch, 16, 14, 14]
        x = self.relu2(x)
        x = self.pool2(x)   # [batch, 16, 7, 7]

        x = x.view(x.size(0), -1)  # [batch, 16*7*7]
        x = self.fc(x)             # [batch, 10]
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_loss_sum = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    train_loss = train_loss_sum / len(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    print(
        f"epoch={epoch} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_mnist_cnn.pth")
        print("已保存当前最佳模型: best_mnist_cnn.pth")

# 用最佳模型在测试集上评估
model.load_state_dict(torch.load("best_mnist_cnn.pth", map_location=device))
test_loss, test_acc = evaluate(test_loader)
print(f"\n最终测试集: loss={test_loss:.4f}, acc={test_acc:.2f}%")