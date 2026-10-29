# Assignment 3, Part 2: train the Part 1 GAN (MNIST_GAN) on MNIST and save it for the API.
#
# Standard GAN training with binary cross entropy and Adam (lr=0.0002,
# betas=(0.5, 0.999)). A grid of generated digits is saved after every epoch
# in checkpoints/mnist_gan_samples/ so you can watch the digits improve.
#
# Run from the project root:  uv run python train_mnist_gan.py
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_mnist_gan
from helper_lib.model import get_model
from helper_lib.generator import generate_samples
from helper_lib.utils import save_model
import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 20

torch.manual_seed(42)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Downloads MNIST (about 12 MB) into data/MNIST the first time
train_loader = get_data_loader('data', batch_size=128, train=True, dataset="MNIST")

gan = get_model("MNIST_GAN")
criterion = nn.BCEWithLogitsLoss()  # binary cross entropy: real = 1, fake = 0
opt_gen = optim.Adam(gan.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(gan.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

train_mnist_gan(gan, train_loader, criterion, (opt_gen, opt_disc), device=device, epochs=EPOCHS,
                sample_dir="checkpoints/mnist_gan_samples")

# Save the weights inside app/ so the FastAPI app (and the Docker image) can load them
save_model(gan.to('cpu'), "app/models/gan_mnist.pth")
generate_samples(gan, 'cpu', num_samples=10)
