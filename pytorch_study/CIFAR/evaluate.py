import argparse
import torch
import torch.nn as nn

from data import get_data_loaders
from model import build_model
from checkpoint import load_checkpoint
from engine import test_with_predictions


CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def compute_confusion_matrix(preds, targets, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for t, p in zip(targets, preds):
        cm[t.item(), p.item()] += 1

    return cm


def compute_per_class_accuracy(confusion_matrix):
    per_class_acc = []

    for i in range(confusion_matrix.size(0)):
        correct = confusion_matrix[i, i].item()
        total = confusion_matrix[i].sum().item()

        if total == 0:
            acc = 0.0
        else:
            acc = 100.0 * correct / total

        per_class_acc.append(acc)

    return per_class_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR10 model')
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='number of dataloader workers')
    parser.add_argument('--data-root', type=str, default='./data', help='dataset root directory')
    args = parser.parse_args()

    device = get_device()
    print('Using device:', device)

    _, _, testloader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()

    model, _, _, _, _ = load_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_model_name=args.model
    )

    test_loss, test_acc, preds, targets = test_with_predictions(
        model, testloader, device, criterion
    )

    print(f'Overall Test Loss: {test_loss:.4f}')
    print(f'Overall Test Acc: {test_acc:.2f}%')

    cm = compute_confusion_matrix(preds, targets, num_classes=len(CLASSES))
    per_class_acc = compute_per_class_accuracy(cm)

    print('\nPer-class Accuracy:')
    for class_name, acc in zip(CLASSES, per_class_acc):
        print(f'{class_name}: {acc:.2f}%')

    print('\nConfusion Matrix:')
    print(cm)


if __name__ == '__main__':
    main()