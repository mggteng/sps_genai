# Usage example for the Module 6 Activity (Adding GAN to Helper Library):
#   - CelebA faces, cropped to 64x64 and scaled to [-1, 1] (as in the Module 6 notebook)
#   - get_model("GAN") returns the generator and the critic together
#   - train_gan trains a Wasserstein GAN with weight clipping (RMSprop, lr=5e-5,
#     5 critic updates per generator update, clip value 0.01)
#   - generate_samples draws new faces from random noise
#
# The notebook trains on all ~160k CelebA images for 20 epochs, which takes many
# hours on a laptop. SUBSET_SIZE and EPOCHS below keep this run short; set
# SUBSET_SIZE = None and EPOCHS = 20 to match the notebook.
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_gan
from helper_lib.model import get_model
from helper_lib.generator import generate_samples
from helper_lib.utils import save_model
import torch
import torch.optim as optim

SUBSET_SIZE = 10000
EPOCHS = 3

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Downloads CelebA (about 1.4 GB) into data/celeba the first time.
# If the download fails, download img_align_celeba yourself and use instead:
#   get_data_loader('data/img_align_celeba', batch_size=64, dataset="CelebAFolder", subset_size=SUBSET_SIZE)
train_loader = get_data_loader('data', batch_size=64, dataset="CelebA", subset_size=SUBSET_SIZE)

gan = get_model("GAN")
opt_gen = optim.RMSprop(gan.generator.parameters(), lr=5e-5)
opt_critic = optim.RMSprop(gan.critic.parameters(), lr=5e-5)

# criterion=None: WGAN uses its own Wasserstein loss inside train_gan
train_gan(gan, train_loader, None, (opt_gen, opt_critic), device=device, epochs=EPOCHS,
          sample_dir="checkpoints/gan_samples")
save_model(gan, "checkpoints/gan.pth")
generate_samples(gan, device, num_samples=10)
