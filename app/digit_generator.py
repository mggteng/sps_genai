"""
MNIST digit generator for the API (Assignment 3, Part 2).

Loads the MNIST GAN from helper_lib with the weights trained by
train_mnist_gan.py, and turns random noise into new hand-written digits.
"""

import io
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from helper_lib.model import get_model

WEIGHTS_PATH = Path(__file__).parent / "models" / "gan_mnist.pth"


class DigitGenerator:
    def __init__(self, weights_path=WEIGHTS_PATH):
        self.model = get_model("MNIST_GAN")
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # BatchNorm uses its saved statistics

    def generate(self, num_images=16, seed=None):
        """Generate `num_images` digits and return them as one PNG grid (bytes)."""
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        noise = torch.randn(num_images, self.model.z_dim, generator=generator)  # (N, 100)

        with torch.no_grad():
            images = self.model.generator(noise)  # (N, 1, 28, 28), pixels in [-1, 1]

        # Put the digits on a grid (up to 8 per row), mapping [-1, 1] back to [0, 1]
        grid = make_grid(images, nrow=min(num_images, 8), padding=2,
                         normalize=True, value_range=(-1, 1))
        buffer = io.BytesIO()
        save_image(grid, buffer, format="png")
        return buffer.getvalue()
