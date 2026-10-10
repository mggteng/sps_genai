import torch
from torchvision import datasets, transforms

# CIFAR-10 class names (from Module 4 Practical 1 and 3)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10"):
    """Create a DataLoader for CIFAR-10 (Module 4) or Fashion-MNIST (Module 5).

    Args:
        data_dir: folder where the dataset is downloaded/stored.
        batch_size: number of images per batch.
        train: True for the training set (shuffled), False for the test set.
        dataset: "CIFAR10" (default, 3x32x32 color images) or
                 "FashionMNIST" (1x32x32 grayscale images, used by the VAE).
    """
    if dataset == "CIFAR10":
        # Prepare the Data (Module 4 Practical 1 and 3)
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset == "FashionMNIST":
        # Module 5 Practical VAE: pad the 28x28 images with zeros to 32x32
        transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
        data = datasets.FashionMNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose CIFAR10 or FashionMNIST")

    # Shuffle the training set only, as in the notebooks
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)
    return loader
