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


def get_model(model_name):
    """Return a new model: one of "FCNN", "CNN", "EnhancedCNN", "CNN64", "VAE", or "GAN"."""
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
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose one of: FCNN, CNN, EnhancedCNN, CNN64, VAE, GAN"
        )
    return model
