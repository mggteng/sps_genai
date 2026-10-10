import torch
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
