# Assignment 2, Part 2: train the Part 1 CNN (CNN64) on CIFAR-10 and save it for the API.
#
# CIFAR-10 images are 32x32, but the Part 1 architecture expects 64x64 inputs,
# so the data loader resizes every image to 64x64 (image_size=64).
#
# Run from the project root:  uv run python train_cifar10_cnn64.py
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.utils import save_model
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

train_loader = get_data_loader('data', batch_size=64, train=True, image_size=64)
test_loader = get_data_loader('data', batch_size=64, train=False, image_size=64)

model = get_model("CNN64")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=10)
evaluate_model(trained_model, test_loader, criterion, device=device)

# Save the weights inside app/ so the FastAPI app (and the Docker image) can load them
save_model(trained_model.to('cpu'), "app/models/cnn64_cifar10.pth")
