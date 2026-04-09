import torch
import torch.nn.functional as F


def train_one_epoch(args, model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"epoch={epoch}, batch={batch_idx}, loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy