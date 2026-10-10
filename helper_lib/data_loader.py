import torch
from torchvision import datasets, transforms

# CIFAR-10 class names (from Module 4 Practical 1 and 3)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10", image_size=None):
    """Create a DataLoader for CIFAR-10 (Module 4) or Fashion-MNIST (Module 5).

    Args:
        data_dir: folder where the dataset is downloaded/stored.
        batch_size: number of images per batch.
        train: True for the training set (shuffled), False for the test set.
        dataset: "CIFAR10" (default, 3x32x32 color images) or
                 "FashionMNIST" (1x32x32 grayscale images, used by the VAE).
        image_size: optional size to resize CIFAR-10 images to, e.g. 64 for the
                    Assignment 2 CNN (CNN64). None keeps the original 32x32.
    """
    if dataset == "CIFAR10":
        # Prepare the Data (Module 4 Practical 1 and 3)
        steps = [transforms.ToTensor()]
        if image_size is not None:
            # Assignment 2: CNN64 expects 64x64 inputs, so upscale CIFAR-10 images
            steps = [transforms.Resize((image_size, image_size))] + steps
        transform = transforms.Compose(steps)
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
