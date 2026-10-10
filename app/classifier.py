"""
CIFAR-10 image classifier for the API (Assignment 2, Part 2).

Loads the CNN64 model from helper_lib with the weights trained by
train_cifar10_cnn64.py, and classifies an uploaded image into one of the
10 CIFAR-10 classes.
"""

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.data_loader import CLASSES
from helper_lib.model import get_model

WEIGHTS_PATH = Path(__file__).parent / "models" / "cnn64_cifar10.pth"

# Same preprocessing as training: resize to 64x64, convert to a tensor in [0, 1]
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class CIFAR10Classifier:
    def __init__(self, weights_path=WEIGHTS_PATH):
        self.model = get_model("CNN64")
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def predict(self, image_bytes, top_k=3):
        """Return the predicted class and the top_k class probabilities."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(image).unsqueeze(0)  # add the batch dimension: 1x3x64x64

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        top_probs, top_idx = torch.topk(probs, top_k)
        return {
            "predicted_class": CLASSES[top_idx[0].item()],
            "top_predictions": [
                {"class": CLASSES[i.item()], "probability": round(p.item(), 4)}
                for p, i in zip(top_probs, top_idx)
            ],
        }
