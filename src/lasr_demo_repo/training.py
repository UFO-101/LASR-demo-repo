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

from lasr_demo_repo.config import TrainingConfig


class MLPAutoencoder(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super(MLPAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.input_dim),
            nn.Sigmoid(),  # Since MNIST pixels are in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(config: TrainingConfig) -> Tuple[MLPAutoencoder, List[float]]:
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root=config.data_dir, train=True, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MLPAutoencoder(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    train_losses = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, data.view(data.size(0), -1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % config.print_every == 0:
                print(
                    f"Epoch: {epoch + 1}/{config.epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}/{config.epochs}, Average Loss: {avg_loss:.6f}")

    return model, train_losses


def visualize_reconstruction(model: MLPAutoencoder, config: TrainingConfig) -> None:
    # Load a batch of test data
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root=config.data_dir, train=False, transform=transform, download=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.vis_batch_size, shuffle=True
    )
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
    # Create config
    config = TrainingConfig(epochs=6)

    # Train the model
    model, losses = train_autoencoder(config)
    # %%

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Visualize some reconstructions
    visualize_reconstruction(model, config)
    visualize_reconstruction(model, config)
