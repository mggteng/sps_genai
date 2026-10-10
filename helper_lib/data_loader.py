import torch
from torchvision import datasets, transforms

# CIFAR-10 class names (from Module 4 Practical 1 and 3)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_data_loader(data_dir, batch_size=32, train=True):
    """Create a CIFAR-10 DataLoader (from Module 4 Practical 1 and 3).

    Args:
        data_dir: folder where CIFAR-10 is downloaded/stored.
        batch_size: number of images per batch.
        train: True for the training set (shuffled), False for the test set.
    """
    # Prepare the Data
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )

    # Shuffle the training set only, as in the notebooks
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
