# Usage example from the Module 5 Activity (section 3), with the missing pieces filled in:
#   - imports for train_vae_model, vae_loss_function and generate_samples
#   - device defined (Apple GPU "mps" if available, as in the notebooks)
#   - Fashion-MNIST dataset (the VAE expects 1x32x32 grayscale images)
#   - criterion = vae_loss_function from the Module 5 notebook
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_vae_model, vae_loss_function
from helper_lib.model import get_model
from helper_lib.generator import generate_samples
import torch
import torch.optim as optim

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

train_loader = get_data_loader('data', batch_size=64, dataset="FashionMNIST")
test_loader = get_data_loader('data', batch_size=64, train=False, dataset="FashionMNIST")

vae = get_model("VAE")
criterion = vae_loss_function
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(vae, train_loader, criterion, optimizer, device=device, epochs=5)
generate_samples(vae, device, num_samples=10)
