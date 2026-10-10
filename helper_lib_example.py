# Usage example from the Module 4 Activity (section 6), plus save_model at the end.
# Note: train and test both use 'data' so CIFAR-10 (170 MB) is downloaded only once.
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.utils import save_model
import torch.nn as nn
import torch.optim as optim

train_loader = get_data_loader('data', batch_size=64)
test_loader = get_data_loader('data', batch_size=64, train=False)

model = get_model("CNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, train_loader, criterion, optimizer, epochs=5)
evaluate_model(trained_model, test_loader, criterion)
save_model(trained_model, "checkpoints/cnn.pth")
