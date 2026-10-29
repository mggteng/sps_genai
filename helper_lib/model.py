import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

NUM_CLASSES = 10


class MLP(nn.Module):
    """Fully Connected Neural Network (from Module 4 Practical 1)."""

    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, NUM_CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


class SimpleCNN(nn.Module):
    """Convolutional Neural Network (from Module 4 Practical 3)."""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )  # Input channels = 3, Output channels = 16
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )  # Input channels = 16, Output channels = 32
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnhancedCNN(nn.Module):
    """CNN with Batch Normalization and Dropout (from Module 4 Practical 3)."""

    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # Convolutional Layer 1 with BatchNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2 with BatchNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after Conv2

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output channels = 64
        self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after Conv3

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output channels = 128
        self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization after Conv4

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Flatten the feature map
        x = x.view(-1, 128 * 2 * 2)

        # Fully connected layer 1 with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)
        return x


class CNN64(nn.Module):
    """CNN for 64x64 RGB images (Assignment 2, Part 1).

    Input:  RGB image 64x64x3
    Conv2D 16 filters, 3x3, stride 1, padding 1 -> ReLU -> MaxPool 2x2, stride 2
    Conv2D 32 filters, 3x3, stride 1, padding 1 -> ReLU -> MaxPool 2x2, stride 2
    Flatten -> Fully connected 100 units -> ReLU -> Fully connected 10 units

    Shapes: 3x64x64 -> 16x64x64 -> 16x32x32 -> 32x32x32 -> 32x16x16
            -> 8192 -> 100 -> 10
    """

    def __init__(self, num_classes=10):
        super(CNN64, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x64x64 -> 16x64x64 -> 16x32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x32x32 -> 32x32x32 -> 32x16x16
        x = x.view(-1, 32 * 16 * 16)          # Flatten: 32x16x16 -> 8192
        x = F.relu(self.fc1(x))               # 8192 -> 100
        x = self.fc2(x)                       # 100 -> 10
        return x


class VariationalEncoder(nn.Module):
    """VAE encoder (from Module 5 Practical VAE).

    Three convolution layers, then two fully connected layers that output the
    mean and log variance of a normal distribution in the latent space.
    """

    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """VAE decoder (from Module 5 Practical VAE).

    A fully connected layer followed by three convolutional transpose layers,
    'reversing' the operations in the encoder.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = torch.sigmoid(self.convtrans3(x))
        return x


class VAE(nn.Module):
    """Variational Autoencoder (from Module 5 Practical VAE)."""

    def __init__(self, encoder, decoder, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


class Critic(nn.Module):
    """WGAN critic (discriminator) for 64x64 RGB images (from Module 6 Practical GAN).

    Outputs a single unbounded score per image (no sigmoid), as required by the
    Wasserstein loss. Shapes: 3x64x64 -> 64x32x32 -> 128x16x16 -> 256x8x8
    -> 512x4x4 -> 1x1x1 -> 1
    """

    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.batchnorm1(self.conv1(x)))
        x = self.act2(self.batchnorm2(self.conv2(x)))
        x = self.act3(self.batchnorm3(self.conv3(x)))
        x = self.act4(self.batchnorm4(self.conv4(x)))
        x = self.conv5(x)
        x = self.flatten(x)
        return x


class Generator(nn.Module):
    """WGAN generator (from Module 6 Practical GAN).

    Maps a random latent vector of size z_dim to a 3x64x64 image with pixel
    values in [-1, 1] (tanh). Shapes: z_dim x1x1 -> 512x4x4 -> 256x8x8
    -> 128x16x16 -> 64x32x32 -> 3x64x64
    """

    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.act4 = nn.ReLU(True)

        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Reshape the latent vector to z_dim x 1 x 1 "image"
        # (the notebook used a lambda for this, which can't be saved with torch.save)
        x = x.view(x.size(0), self.z_dim, 1, 1)
        x = self.act1(self.bn1(self.deconv1(x)))
        x = self.act2(self.bn2(self.deconv2(x)))
        x = self.act3(self.bn3(self.deconv3(x)))
        x = self.act4(self.bn4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x


class GAN(nn.Module):
    """Wasserstein GAN (from Module 6 Practical GAN).

    Holds the generator and the critic together so get_model("GAN") can return
    a single model. Calling the GAN on latent vectors runs the generator.
    """

    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)


# ---------------------------------------------------------------------------
# GAN for MNIST hand-written digits (Assignment 3, Part 1)
# ---------------------------------------------------------------------------

class MNISTGenerator(nn.Module):
    """Generator for 28x28 MNIST digits (Assignment 3).

    Input: noise vector of shape (batch size, 100)
    Fully connected 100 -> 7*7*128, then reshape to 128x7x7
    ConvTranspose2d 128 -> 64, kernel 4, stride 2, padding 1 -> 64x14x14,
        then BatchNorm2d and ReLU
    ConvTranspose2d 64 -> 1, kernel 4, stride 2, padding 1 -> 1x28x28,
        then Tanh (pixels in [-1, 1])
    """

    def __init__(self, z_dim=100):
        super(MNISTGenerator, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = z.view(z.size(0), self.z_dim)    # also accepts noise shaped (B, 100, 1, 1)
        x = self.fc(z)                        # (B, 100) -> (B, 6272)
        x = x.view(-1, 128, 7, 7)             # reshape -> (B, 128, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))  # -> (B, 64, 14, 14)
        x = torch.tanh(self.deconv2(x))       # -> (B, 1, 28, 28)
        return x


class MNISTDiscriminator(nn.Module):
    """Discriminator for 28x28 MNIST digits (Assignment 3).

    Input: image of shape (1, 28, 28)
    Conv2d 1 -> 64, kernel 4, stride 2, padding 1 -> 64x14x14, then LeakyReLU(0.2)
    Conv2d 64 -> 128, kernel 4, stride 2, padding 1 -> 128x7x7,
        then BatchNorm2d and LeakyReLU(0.2)
    Flatten -> Linear 6272 -> 1

    The output is one score per image (a logit). sigmoid(score) is the
    probability that the image is real; the sigmoid is applied inside the loss
    (nn.BCEWithLogitsLoss), which is the numerically stable way to train it.
    """

    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)            # (B, 1, 28, 28) -> (B, 64, 14, 14)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)  # -> (B, 128, 7, 7)
        x = self.flatten(x)                              # -> (B, 6272)
        return self.fc(x)                                # -> (B, 1)


class MNISTGAN(nn.Module):
    """Generator and discriminator together, so get_model("MNIST_GAN") returns one model.
    Calling the MNISTGAN on noise runs the generator."""

    def __init__(self, z_dim=100):
        super(MNISTGAN, self).__init__()
        self.z_dim = z_dim
        self.generator = MNISTGenerator(z_dim)
        self.discriminator = MNISTDiscriminator()

    def forward(self, z):
        return self.generator(z)


# ---------------------------------------------------------------------------
# Diffusion model (from Module 8 Practical 2 - Diffusion Methods)
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Encodes the noise variance as sine/cosine features (Module 8 Practical 2)."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    """
    diffusion_times: Tensor of shape (T,) with values in [0, 1)
    Returns:
        noise_rates: Tensor of shape (T,)
        signal_rates: Tensor of shape (T,)
    """
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    # diffusion_times: Tensor of shape [T] or [B] with values in [0, 1]
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # Flatten diffusion_times to handle any shape
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    # Compute start and end angles from signal rate bounds
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    # Linearly interpolate angles
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    # Compute signal and noise rates
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


class ResidualBlock(nn.Module):
    """UNet residual block: conv -> swish -> conv, plus a skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        # (The notebook also defined a BatchNorm2d here but never used it in
        # forward(), so it is left out.)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """UNet down block: residual blocks (saved as skips), then halve the size."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """UNet up block: double the size, then residual blocks joined with the skips."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """UNet that predicts the noise in a noisy image (Module 8 Practical 2).

    Inputs: noisy images (B, C, H, W) and noise variances (B, 1, 1, 1).
    Output: predicted noise with the same shape as the images.
    """

    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        # (The notebook also defined an embedding_proj layer that forward() never
        # used, so it is left out.)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # Keep zero init like TF reference

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size like TF reference
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    """Denoising diffusion model (Module 8 Practical 2).

    Wraps the UNet (`network`) and an exponential moving average copy of it
    (`ema_network`) that is used to generate images. Images are normalized to
    mean 0 and variance 1 before noise is added.

    Changes from the notebook:
      - normalizer_mean / normalizer_std are buffers (one value per color
        channel), so they move to the GPU with .to(device) and are saved with
        the weights by save_model. train_diffusion fills them from the data.
      - the debug print() calls were removed to keep the training output short.
    """

    def __init__(self, model, schedule_fn, ema_decay=0.8):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        for p in self.ema_network.parameters():
            p.requires_grad_(False)  # the EMA copy is updated by hand, not by the optimizer
        self.ema_decay = ema_decay
        self.schedule_fn = schedule_fn
        self.register_buffer("normalizer_mean", torch.zeros(1, model.num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, model.num_channels, 1, 1))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    def set_normalizer(self, mean, std):
        self.normalizer_mean.copy_(torch.as_tensor(mean).reshape(self.normalizer_mean.shape))
        self.normalizer_std.copy_(torch.as_tensor(std).reshape(self.normalizer_std.shape))
        self.normalizer_fitted.fill_(True)

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Use EMA network for inference, main network for training
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=None, initial_noise=None):
        if image_size is None:
            image_size = self.network.image_size
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)

        return loss.item()

    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)

        return loss.item()

    def forward(self, num_images=10, diffusion_steps=100):
        # Calling the model generates images, like calling the GAN runs its generator
        return self.generate(num_images, diffusion_steps)


def get_model(model_name):
    """Return a new model: one of "FCNN", "CNN", "EnhancedCNN", "CNN64", "VAE", "GAN", "MNIST_GAN", or "Diffusion".

    "Diffusion" is a UNet for 64x64 RGB images wrapped in a DiffusionModel with
    the offset cosine diffusion schedule (Module 8 Practical 2).
    """
    if model_name == "FCNN":
        model = MLP()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "CNN64":
        model = CNN64()
    elif model_name == "VAE":
        model = VAE(VariationalEncoder(), Decoder())
    elif model_name == "GAN":
        model = GAN(z_dim=100)
    elif model_name == "MNIST_GAN":
        model = MNISTGAN(z_dim=100)
    elif model_name == "Diffusion":
        unet = UNet(image_size=64, num_channels=3)
        model = DiffusionModel(unet, offset_cosine_diffusion_schedule)
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose one of: FCNN, CNN, EnhancedCNN, CNN64, VAE, GAN, MNIST_GAN, Diffusion"
        )
    return model
