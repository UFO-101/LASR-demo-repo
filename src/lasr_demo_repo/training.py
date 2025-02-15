# %%
#!%load_ext autoreload
#!%autoreload 2

from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128) -> None:
        super(MLPAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid(),  # Since MNIST pixels are in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
    epochs: int = 20, batch_size: int = 128, learning_rate: float = 1e-3
) -> Tuple[MLPAutoencoder, List[float]]:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./.data", train=True, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MLPAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, data.view(data.size(0), -1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

    return model, train_losses


def visualize_reconstruction(model: MLPAutoencoder, device: str = "cuda") -> None:
    # Load a batch of test data
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root="./.data", train=False, transform=transform, download=True
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    data, _ = next(iter(test_loader))

    # Get reconstructions
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))

    for i in range(8):
        # Original images
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")

        # Reconstructed images
        axes[1, i].imshow(reconstructions[i].cpu().view(28, 28), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train the model
    model, losses = train_autoencoder(epochs=6)
    # %%

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Visualize some reconstructions
    visualize_reconstruction(model)
    visualize_reconstruction(model)
