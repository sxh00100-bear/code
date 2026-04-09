import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device =", device)

batch_size = 64
learning_rate = 0.001
epochs = 5

transform = transforms.ToTensor()

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = self.conv1(x)   # -> [batch, 8, 28, 28]
        x = self.relu1(x)
        x = self.pool1(x)   # -> [batch, 8, 14, 14]

        x = self.conv2(x)   # -> [batch, 16, 14, 14]
        x = self.relu2(x)
        x = self.pool2(x)   # -> [batch, 16, 7, 7]

        x = x.view(x.size(0), -1)   # -> [batch, 16*7*7]
        x = self.fc(x)              # -> [batch, 10]
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"epoch={epoch}, batch={batch_idx}, loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"epoch={epoch}, 平均loss={avg_loss:.4f}")


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
    
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("模型已保存到 mnist_cnn.pth")