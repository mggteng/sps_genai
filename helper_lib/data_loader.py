import os

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

# CIFAR-10 class names (from Module 4 Practical 1 and 3)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CustomImageDataset(Dataset):
    """Images from a folder, e.g. a manually downloaded img_align_celeba
    (from Module 6 Practical GAN). Returns (image, 0): the label is not used by GANs.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label (not used in GANs)


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10", image_size=None,
                    subset_size=None):
    """Create a DataLoader for CIFAR-10 (Module 4), Fashion-MNIST (Module 5) or CelebA (Module 6).

    Args:
        data_dir: folder where the dataset is downloaded/stored.
        batch_size: number of images per batch.
        train: True for the training set (shuffled), False for the test set.
        dataset: "CIFAR10" (default, 3x32x32 color images),
                 "FashionMNIST" (1x32x32 grayscale images, used by the VAE),
                 "CelebA" (3x64x64 face images scaled to [-1, 1], used by the GAN;
                 downloaded with torchvision, about 1.4 GB), or
                 "CelebAFolder" (same images from a folder you downloaded yourself;
                 data_dir is that folder, e.g. "data/img_align_celeba").
        image_size: optional size to resize CIFAR-10 images to, e.g. 64 for the
                    Assignment 2 CNN (CNN64). None keeps the original 32x32.
        subset_size: optional number of images to keep (the first N), for quick
                     runs on a laptop. None uses the whole dataset.
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
    elif dataset in ("CelebA", "CelebAFolder"):
        # Module 6 Practical GAN: crop faces to 64x64 and normalize pixels to
        # [-1, 1] to match the tanh output of the generator
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        if dataset == "CelebA":
            split = "train" if train else "test"
            data = datasets.CelebA(root=data_dir, split=split, download=True, transform=transform)
        else:
            data = CustomImageDataset(root_dir=data_dir, transform=transform)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose CIFAR10, FashionMNIST, CelebA or CelebAFolder"
        )

    if subset_size is not None:
        data = Subset(data, range(min(subset_size, len(data))))

    # Shuffle the training set only, as in the notebooks
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)
    return loader
