import torch
from torchvision import datasets, transforms

from model import Net


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    model = Net().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    image, label = test_dataset[0]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    print(f"真实类别: {label} ({classes[label]})")
    print(f"预测类别: {pred} ({classes[pred]})")


if __name__ == "__main__":
    main()