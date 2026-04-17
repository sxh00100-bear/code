import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset, random_split


def get_data_loaders(batch_size, num_workers, data_root):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform_test
    )

    train_size = 45000
    val_size = 5000
    generator = torch.Generator().manual_seed(42)
    train_subset_indices, val_subset_indices = random_split(
        range(len(full_train_dataset)),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform_test
    )

    train_dataset = Subset(train_dataset, train_subset_indices.indices)
    val_dataset = Subset(val_dataset, val_subset_indices.indices)

    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return trainloader, valloader, testloader
