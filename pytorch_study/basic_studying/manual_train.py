import torch

# 训练数据
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])

# 要学习的参数
w = torch.randn(1, requires_grad=True)

# 学习率
lr = 0.01

for epoch in range(100):
    # forward
    y_pred = w * x

    # loss
    loss = ((y_pred - y_true) ** 2).mean()

    # backward
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= lr * w.grad

    # 清空梯度
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch={epoch}, loss={loss.item():.4f}, w={w.item():.4f}")



