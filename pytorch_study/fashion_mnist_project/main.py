import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data import get_data_loaders
from model import Net
from engine import train_one_epoch, evaluate

def main():
    # Training settings, you can specify them in the command line or keep default
    parser = argparse.ArgumentParser(description='PyTorch FashionMNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if args.save_model:
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pt")
                print(f"Best model saved at epoch {epoch}, val_acc={val_acc:.2f}%")
    if args.save_model:
        model.load_state_dict(torch.load("best_model.pt"))
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    else:
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
