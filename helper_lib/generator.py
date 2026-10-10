import math

import matplotlib.pyplot as plt
import torch


def generate_samples(model, device, num_samples=10, latent_dim=2):
    """Generate new images with a trained VAE (from Module 5 Practical VAE).

    Samples `num_samples` points from a standard normal distribution in the
    latent space, runs the decoder to construct images, and plots them on a grid.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()

    # Arrange the samples on a grid (up to 5 images per row)
    cols = min(num_samples, 5)
    rows = math.ceil(num_samples / cols)
    _fig, _axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for _i, _ax in enumerate(_axes.flat):
        if _i < num_samples:
            _ax.imshow(samples[_i].squeeze(), cmap='gray')
        _ax.axis('off')

    plt.tight_layout()
    plt.show()
