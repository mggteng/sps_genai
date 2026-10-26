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


# ---------------------------------------------------------------------------
# Diffusion model (from Module 8 Practical 2 - Diffusion Methods)
# ---------------------------------------------------------------------------

def fit_diffusion_normalizer(model, data_loader, device='cpu', max_batches=50):
    """Measure the mean and standard deviation of each color channel of the
    training images, so the diffusion model can scale them to mean 0 and
    variance 1 (the notebook assumes the images are preprocessed this way)."""
    total, total_sq, count = 0.0, 0.0, 0
    for batch_number, (images, _) in enumerate(data_loader):
        if batch_number >= max_batches:
            break
        images = images.to(device)
        total = total + images.sum(dim=(0, 2, 3))
        total_sq = total_sq + (images ** 2).sum(dim=(0, 2, 3))
        count += images.size(0) * images.size(2) * images.size(3)
    mean = total / count
    std = (total_sq / count - mean ** 2).clamp(min=1e-8).sqrt()
    model.set_normalizer(mean, std)
    print(f"Normalizer: mean={[round(m, 3) for m in mean.tolist()]}, std={[round(s, 3) for s in std.tolist()]}")


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
                    val_loader=None, checkpoint_dir=None, sample_dir=None):
    """Diffusion training loop (from Module 8 Practical 2). Returns the trained model.

    Each step picks a random diffusion time for every image, adds that much
    noise, and trains the UNet to predict the noise (model.train_step).

    Args:
        model: a DiffusionModel from get_model("Diffusion").
        data_loader: training images with pixels in [0, 1], e.g.
            get_data_loader('data', batch_size=64, dataset="Flowers102").
        criterion: loss between predicted and real noise, e.g. nn.L1Loss()
            (mean absolute error, as in the TF reference the notebook follows)
            or nn.MSELoss(). If None, nn.L1Loss() is used.
        optimizer: optimizer for model.network, e.g.
            torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4).
        val_loader: optional validation images; if given, the average
            validation loss is printed after every epoch (model.test_step).
        checkpoint_dir: optional folder; if given, a checkpoint is saved after
            every epoch (diffusion_epoch_001.pth, ...) plus diffusion_best.pth,
            as in the notebook.
        sample_dir: optional folder; if given, a grid of generated images is
            saved after every epoch (the notebook showed them with plt.show()).

    The notebook's version took (train_loader, val_loader, optimizer, loss_fn);
    here the arguments follow the activity's train_diffusion signature, and
    val_loader is optional.
    """
    if criterion is None:
        criterion = torch.nn.L1Loss()
    model.to(device)

    # The images must have mean 0 and variance 1 before noise is added
    if not bool(model.normalizer_fitted):
        fit_diffusion_normalizer(model, data_loader, device)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_losses = []
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            train_losses.append(loss)
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')

        avg_train_loss = sum(train_losses) / len(train_losses)
        model.eval()

        avg_val_loss = None
        if val_loader is not None:
            val_losses = []
            for images, _ in tqdm(val_loader, ncols=120, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(device)
                val_losses.append(model.test_step(images, criterion))
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # Save checkpoint every epoch, and the best one so far (as in the notebook)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),  # UNet, EMA UNet and normalizer
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1:03d}.pth')
            torch.save(checkpoint, checkpoint_path)
            epoch_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'diffusion_best.pth'))
                print(f"New best model saved at epoch {epoch+1} with loss: {epoch_loss:.4f}")

        # Save generated samples (the notebook's plot_images used 20 diffusion steps)
        if sample_dir is not None:
            os.makedirs(sample_dir, exist_ok=True)
            samples = model.generate(num_images=18, diffusion_steps=20).cpu()
            save_image(make_grid(samples, nrow=6), os.path.join(sample_dir, f"diffusion_epoch_{epoch+1:03d}.png"))

    print("Finished Training")
    return model
