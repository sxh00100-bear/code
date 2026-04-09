import torch
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

# 原始数据
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

dataset = MyDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print("len(dataset) =", len(dataset))
print("dataset[0] =", dataset[0])
print()

for batch_idx, (x, y) in enumerate(dataloader):
    print(f"batch {batch_idx}")
    print("x =", x)
    print("x.shape =", x.shape)
    print("y =", y)
    print("y.shape =", y.shape)
    print()