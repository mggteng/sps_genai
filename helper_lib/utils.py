import os

import torch


def save_model(model, path="model.pth"):
    """Save only the model weights (state_dict) to `path`."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path


# Checkpoint helpers (from Module 4 Practical 1)

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints_fcnn'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'fcnn_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy
