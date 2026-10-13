import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """Training loop (from Module 4 Practical 1 and 3). Returns the trained model."""
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix(
                    {
                        "avg accuracy": f"{running_correct/running_total:.3f}",
                        "avg loss": f"{running_loss/(batch_number+1):.4f}",
                    }
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * running_correct / running_total
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

    print("Finished Training")
    return model


# ---------------------------------------------------------------------------
# Variational Autoencoder (from Module 5 Practical VAE)
# ---------------------------------------------------------------------------

def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss: reconstruction error (BCE) plus KL divergence.

    The KL term pushes the latent distribution toward a standard normal
    distribution (mean 0, variance 1).
    """
    beta = 500
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * BCE + KLD


def train_vae_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """VAE training loop (from Module 5 Practical VAE). Returns the trained model.

    `criterion` is called as criterion(reconstruction, inputs, mu, logvar),
    for example vae_loss_function above.
    """
    model.to(device)

    # Train the model
    model.train()
    for epoch in range(epochs):
        train_loader_with_progress = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data in train_loader_with_progress:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = criterion(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loader_with_progress.set_postfix(loss=f'{loss.item():.4f}')

    print("Finished Training")
    return model


# ---------------------------------------------------------------------------
# Wasserstein GAN (from Module 6 Practical GAN)
# ---------------------------------------------------------------------------

def train_gan(model, data_loader, criterion=None, optimizer=None, device='cpu', epochs=10,
              n_critic=5, clip_value=0.01, lr=5e-5, sample_dir=None):
    """WGAN training loop (from Module 6 Practical GAN). Returns the trained model.

    Args:
        model: a GAN from get_model("GAN") (has .generator, .critic and .z_dim).
        data_loader: real images scaled to [-1, 1], e.g.
            get_data_loader('data', dataset="CelebA").
        criterion: not used. WGAN computes its own Wasserstein loss
            (critic: -(D(real) - D(fake)), generator: -D(fake)), so pass None.
            The argument is kept so train_gan has the same signature as the
            other trainers in helper_lib.
        optimizer: a tuple (opt_gen, opt_critic), because the generator and
            the critic are trained separately. If None, two RMSprop optimizers
            with learning rate `lr` are created, as in the notebook.
        n_critic: critic updates per generator update (notebook: 5).
        clip_value: weight clipping range for the critic to keep it Lipschitz
            (notebook: 0.01).
        sample_dir: optional folder; if given, a grid of 64 generated images is
            saved there after every epoch (gan_epoch_001.png, ...).
    """
    gen = model.generator.to(device)
    critic = model.critic.to(device)
    z_dim = model.z_dim

    if optimizer is None:
        opt_gen = torch.optim.RMSprop(gen.parameters(), lr=lr)
        opt_critic = torch.optim.RMSprop(critic.parameters(), lr=lr)
    else:
        opt_gen, opt_critic = optimizer

    # Fixed noise so the saved sample grids show the same "faces" improving over time
    fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)

    for epoch in range(epochs):
        gen.train()
        critic.train()
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (real, _) in enumerate(train_loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            ## === Train Critic === ##
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise).detach()
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            ## === Train Generator === ##
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix(
                    {
                        "D loss": f"{loss_critic.item():.4f}",
                        "G loss": f"{loss_gen.item():.4f}",
                    }
                )

        print(f"Epoch {epoch+1}: D loss={loss_critic.item():.4f}, G loss={loss_gen.item():.4f}")

        # Save sample images (the notebook showed them with plt.show() instead)
        if sample_dir is not None:
            os.makedirs(sample_dir, exist_ok=True)
            gen.eval()
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            grid = make_grid(fake, normalize=True)
            save_image(grid, os.path.join(sample_dir, f"gan_epoch_{epoch+1:03d}.png"))

    print("Finished Training")
    return model
