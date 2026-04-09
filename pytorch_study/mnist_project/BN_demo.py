import torch
import torch.nn as nn

torch.manual_seed(42)

bn = nn.BatchNorm1d(num_features=4)

x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0]
])

print("输入 x =")
print(x)
print()

# 训练模式
bn.train()
print("=== train mode ===")
for i in range(3):
    y = bn(x)
    print(f"第{i+1}次输出 y =")
    print(y)
    print()

# 评估模式
bn.eval()
print("=== eval mode ===")
for i in range(3):
    y = bn(x)
    print(f"第{i+1}次输出 y =")
    print(y)
    print()