import math

import matplotlib.pyplot as plt
import torch


def generate_samples(model, device, num_samples=10, diffusion_steps=100, latent_dim=2):
    """Generate new images with a trained VAE, GAN or diffusion model and plot them on a grid.

    - VAE (Module 5 Practical VAE): samples `num_samples` points from a standard
      normal distribution in the 2-D latent space and runs the decoder.
    - GAN (Module 6 Practical GAN): samples `num_samples` random noise vectors of
      size model.z_dim (100) and runs the generator. `latent_dim` is ignored
      because the GAN knows its own latent size.
    - Diffusion (Module 8 Practical 2): samples `num_samples` images of pure
      noise from a standard normal distribution and runs the reverse diffusion
      for `diffusion_steps` steps (more steps: slower but cleaner images).
    """
    model.to(device)
    model.eval()
    is_gan = hasattr(model, "generator")
    is_diffusion = hasattr(model, "reverse_diffusion")
    is_color = is_gan or is_diffusion

    with torch.no_grad():
        if is_diffusion:
            # Standard normal noise -> reverse diffusion -> pixels in [0, 1]
            initial_noise = torch.randn(
                num_samples, model.network.num_channels,
                model.network.image_size, model.network.image_size,
            ).to(device)
            samples = model.generate(num_samples, diffusion_steps, initial_noise=initial_noise)
        elif is_gan:
            z = torch.randn(num_samples, model.z_dim, 1, 1).to(device)
            samples = model.generator(z)
            # Undo the [-1, 1] normalization (tanh output) back to [0, 1]
            samples = (samples * 0.5) + 0.5
            samples = samples.clamp(0, 1)
        else:
            z = torch.randn(num_samples, latent_dim).to(device)
            samples = model.decoder(z)
        samples = samples.cpu()

    # Arrange the samples on a grid (up to 5 images per row)
    cols = min(num_samples, 5)
    rows = math.ceil(num_samples / cols)
    _fig, _axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for _i, _ax in enumerate(_axes.flat):
        if _i < num_samples:
            if is_color:
                # imshow needs height x width x channels for color images
                _ax.imshow(samples[_i].permute(1, 2, 0).numpy())
            else:
                _ax.imshow(samples[_i].squeeze().numpy(), cmap='gray')
        _ax.axis('off')

    plt.tight_layout()
    plt.show()
