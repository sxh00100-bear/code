import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = SimpleLinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    for x, y in dataloader:
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1000 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"epoch={epoch}, loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")

