import torch
import torch.nn.functional as F
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
