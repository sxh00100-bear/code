import torch
import torch.nn as nn

torch.manual_seed(42)

bn = nn.BatchNorm1d(num_features=4)

x1 = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0]
])

x2 = torch.tensor([
    [10.0, 20.0, 30.0, 40.0],
    [11.0, 21.0, 31.0, 41.0],
    [12.0, 22.0, 32.0, 42.0]
])

print("初始 running_mean =", bn.running_mean)
print("初始 running_var  =", bn.running_var)
print()

bn.train()

y1 = bn(x1)
print("经过 x1 后:")
print("running_mean =", bn.running_mean)
print("running_var  =", bn.running_var)
print()

y2 = bn(x2)
print("经过 x2 后:")
print("running_mean =", bn.running_mean)
print("running_var  =", bn.running_var)
print()

bn.eval()
y3 = bn(x2)
print("eval 模式下输出 y3 =")
print(y3)
#Dropout
##训练时随机把一部分神经元输出置 0，缓解过拟合；测试时关闭。

#BatchNorm
#对某一层的输出特征做标准化。
#训练时用当前 batch 的统计量；测试时用训练过程中累计的统计量。