import torch

from data import get_dataloaders
from model import Net
from engine import train_one_epoch, evaluate


def main():
    device = torch.device("cpu")
    train_loader, test_loader = get_dataloaders(batch_size=64)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    num_epochs = 5

    # 如果有 checkpoint，就恢复
    checkpoint_path = "checkpoint.pth"
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resume from epoch {start_epoch}")
    except FileNotFoundError:
        print("No checkpoint found, start from scratch")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}")

        train_one_epoch(model, train_loader, optimizer, device)
        evaluate(model, test_loader, device)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)


if __name__ == "__main__":
    main()