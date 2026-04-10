import torch
import os

def train(model, trainloader, device, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = train_loss / len(trainloader)
    train_acc = 100. * correct / total
    return avg_loss, train_acc


def evaluate(model, valloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in valloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_loss = total_loss / len(valloader)
    val_acc = 100.0 * correct / total

    return avg_loss, val_acc





def test(model, testloader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = 100.*correct/total
    avg_test_loss = test_loss / len(testloader)
    print(f'Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    return avg_test_loss, test_acc


def test_with_predictions(model, testloader, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

    avg_test_loss = test_loss / len(testloader)
    test_acc = 100.0 * correct / total

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return avg_test_loss, test_acc, all_preds, all_targets