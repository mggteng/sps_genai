# Usage example for the Module 8 Activity (Adding Diffusion to Helper Library):
#   - Oxford Flowers 102, cropped to 64x64 color images
#   - get_model("Diffusion") returns a UNet wrapped in a DiffusionModel
#     (offset cosine diffusion schedule, EMA copy of the UNet for generation)
#   - train_diffusion trains the UNet to predict the noise added to the images
#   - generate_samples starts from standard normal noise and runs the reverse
#     diffusion to build new images
#
# Diffusion models need many epochs before the samples look like flowers.
# EPOCHS below keeps this run short on a laptop; increase it (e.g. 50) for
# better images.
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_diffusion
from helper_lib.model import get_model
from helper_lib.generator import generate_samples
from helper_lib.utils import save_model
import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 10

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Downloads Oxford Flowers 102 (about 330 MB) into data/flowers-102 the first time.
train_loader = get_data_loader('data', batch_size=64, dataset="Flowers102")
val_loader = get_data_loader('data', batch_size=64, train=False, dataset="Flowers102")

diffusion = get_model("Diffusion")
optimizer = optim.AdamW(diffusion.network.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.L1Loss()  # mean absolute error between predicted and real noise

train_diffusion(diffusion, train_loader, criterion, optimizer, device=device, epochs=EPOCHS,
                val_loader=val_loader, checkpoint_dir="checkpoints/diffusion",
                sample_dir="checkpoints/diffusion_samples")
save_model(diffusion, "checkpoints/diffusion.pth")
generate_samples(diffusion, device, num_samples=10, diffusion_steps=100)
