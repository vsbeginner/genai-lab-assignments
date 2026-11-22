"""Laboratory Assignment 02: Generative Adversarial Networks (GANs) for Image Synthesis.

Course: ETMMGA304 - Generative AI and Its Applications
Institution: K.R. Mangalam University (Master of Computer Applications)
Curriculum Reference: BTX 3 (Apply), BTX 4 (Analyze) & BTX 5 (Evaluate)

This script implements a canonical Generative Adversarial Network (GAN) trained on
the Fashion-MNIST benchmark (60,000 grayscale clothing items normalized to [-1, 1]).
It demonstrates:
  1. Deep multi-layer generator mapping Gaussian latent noise to image space via Tanh.
  2. Binary classification discriminator predicting real vs synthesized authenticity logits.
  3. Alternating minimax optimization with gradient detachment (preventing discriminator
     loss from backpropagating into generator).
  4. Manifold tracking across epochs 1, 5, and 10 using invariant fixed noise vectors.

Outputs:
  - lab_outputs/fashion_generator.pt: Serialized PyTorch state dictionary for generator.
  - lab_outputs/gan_epoch_01.png: Synthesized 5x5 grid at Epoch 1 (early noise).
  - lab_outputs/gan_epoch_05.png: Synthesized 5x5 grid at Epoch 5 (intermediate shapes).
  - lab_outputs/gan_epoch_10.png: Synthesized 5x5 grid at Epoch 10 (converged garments).
  - lab_outputs/gan_losses.png: Adversarial loss curves (Discriminator vs Generator).
  - lab_outputs/results.json: Per-epoch loss telemetry across 10 epochs.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid


# -----------------------------------------------------------------------------
# Configuration & Hyperparameters
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42
BATCH_SIZE: int = 128
LATENT_DIM: int = 64
LEARNING_RATE: float = 0.0002
ADAM_BETAS: Tuple[float, float] = (0.5, 0.999)
TOTAL_EPOCHS: int = 10
GRID_SAMPLES: int = 25


def configure_environment(seed: int = RANDOM_SEED) -> Tuple[torch.device, Path]:
    """Configures deterministic seed environment and output directories."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("lab_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return device, output_dir


# -----------------------------------------------------------------------------
# Network Architectures
# -----------------------------------------------------------------------------
class Generator(nn.Module):
    """Deep Multi-Layer Perceptron Generator for Fashion-MNIST synthesis."""

    def __init__(self, latent_dim: int = LATENT_DIM, output_dim: int = 784) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh(),  # Bounded to [-1, 1] matching normalized image space
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Discriminator(nn.Module):
    """Deep Multi-Layer Perceptron Discriminator for authenticity classification."""

    def __init__(self, input_dim: int = 784) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # Unbounded logit output for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------------------------------------------------------
# Dataset Ingestion
# -----------------------------------------------------------------------------
def load_fashion_mnist(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, int]:
    """Ingests Fashion-MNIST dataset and scales pixel values to [-1, 1]."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = str(project_root / "data")

    # Scale [0, 1] to [-1, 1] to align with Generator's Tanh activation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, len(dataset)


# -----------------------------------------------------------------------------
# Visualization Utilities
# -----------------------------------------------------------------------------
def save_sample_grid(
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    epoch: int,
    output_dir: Path,
) -> None:
    """Renders and serializes fixed-noise synthesized samples to disk."""
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu().reshape(-1, 1, 28, 28)
    generator.train()

    # Rescale [-1, 1] back to [0, 1] for visualization
    grid = make_grid((fake_images + 1.0) / 2.0, nrow=5)
    plt.figure(figsize=(5.0, 5.0))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(f"Synthesized Fashion Items - Epoch {epoch:02d}")
    plt.axis("off")
    output_path = output_dir / f"gan_epoch_{epoch:02d}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_loss_curves(history: List[Tuple[float, float]], output_path: Path) -> None:
    """Plots and serializes adversarial training loss trajectories."""
    epochs = range(1, len(history) + 1)
    d_losses = [d for d, _ in history]
    g_losses = [g for _, g in history]

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(epochs, d_losses, label="Discriminator Loss ($D$)", color="#1f77b4", linewidth=2.0)
    plt.plot(epochs, g_losses, label="Generator Loss ($G$)", color="#ff7f0e", linewidth=2.0)
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Batch Binary Cross-Entropy Loss")
    plt.title("Adversarial Loss Convergence (Fashion-MNIST)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Alternating Training Step
# -----------------------------------------------------------------------------
def train_adversarial_batch(
    real_images: torch.Tensor,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_d: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
    loss_fn: nn.Module,
    latent_dim: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Executes one alternating Minimax step between Discriminator and Generator."""
    batch_size = real_images.size(0)

    # -------------------------------------------------------------------------
    # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
    # -------------------------------------------------------------------------
    noise = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(noise)

    real_logits = discriminator(real_images)
    # Detach fake images to prevent gradients from flowing into Generator weights
    fake_logits = discriminator(fake_images.detach())

    loss_real = loss_fn(real_logits, torch.ones_like(real_logits))
    loss_fake = loss_fn(fake_logits, torch.zeros_like(fake_logits))
    d_loss = (loss_real + loss_fake) / 2.0

    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # -------------------------------------------------------------------------
    # 2. Update Generator: maximize log(D(G(z))) -> minimize BCE with target=1
    # -------------------------------------------------------------------------
    # Temporarily freeze Discriminator gradients to conserve memory
    discriminator.requires_grad_(False)
    opt_g.zero_grad()

    # Re-evaluate synthesized batch through Discriminator
    re_evaluated_logits = discriminator(fake_images)
    g_loss = loss_fn(re_evaluated_logits, torch.ones_like(re_evaluated_logits))

    g_loss.backward()
    opt_g.step()

    discriminator.requires_grad_(True)

    return d_loss.item(), g_loss.item()


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    device, output_dir = configure_environment(RANDOM_SEED)
    print("=" * 70)
    print(f"ETMMGA304 Lab 02: Generative Adversarial Network Execution | Device: {device}")
    print("=" * 70)

    # Step 1: Ingest Fashion-MNIST
    print("[1/4] Ingesting & normalizing Fashion-MNIST dataset...")
    loader, dataset_size = load_fashion_mnist(BATCH_SIZE)

    # Step 2: Initialize Networks & Optimizers
    print("[2/4] Instantiating Generator & Discriminator networks...")
    generator = Generator(latent_dim=LATENT_DIM, output_dim=784).to(device)
    discriminator = Discriminator(input_dim=784).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    # Fixed latent vectors for deterministic epoch-by-epoch evaluation
    fixed_noise = torch.randn(GRID_SAMPLES, LATENT_DIM, device=device)

    # Checkpoint shape & range assertions
    sample_real, _ = next(iter(loader))
    with torch.no_grad():
        test_fake = generator(torch.randn(sample_real.size(0), LATENT_DIM, device=device))
        test_logits = discriminator(sample_real.to(device))
    print(f"      Real Image Range: [{sample_real.min():.2f}, {sample_real.max():.2f}]")
    print(f"      Generator Output Range: [{test_fake.min():.2f}, {test_fake.max():.2f}] (Shape: {tuple(test_fake.shape)})")
    print(f"      Discriminator Output Shape: {tuple(test_logits.shape)}")
    assert test_fake.shape[1] == 784 and test_logits.shape[1] == 1

    # Step 3 & 4: Adversarial Training Loop
    print(f"\n[3/4] Commencing alternating minimax training ({TOTAL_EPOCHS} epochs)...")
    history: List[Tuple[float, float]] = []

    for epoch in range(1, TOTAL_EPOCHS + 1):
        batch_losses = []
        for real_batch, _ in loader:
            d_l, g_l = train_adversarial_batch(
                real_batch.to(device),
                generator,
                discriminator,
                opt_d,
                opt_g,
                loss_fn,
                LATENT_DIM,
                device,
            )
            batch_losses.append((d_l, g_l))

        avg_d = sum(d for d, _ in batch_losses) / len(batch_losses)
        avg_g = sum(g for _, g in batch_losses) / len(batch_losses)
        history.append((avg_d, avg_g))

        print(f"      Epoch [{epoch:02d}/{TOTAL_EPOCHS:02d}] -> Discriminator Loss: {avg_d:.4f} | Generator Loss: {avg_g:.4f}")

        # Checkpoint sample grid at epochs 1, 5, and 10
        if epoch in (1, 5, 10):
            print(f"      ==> Serializing fixed-noise sample grid for Epoch {epoch:02d}...")
            save_sample_grid(generator, fixed_noise, epoch, output_dir)

    # Serialize Artifacts
    print("\n[4/4] Finalizing telemetry and saving model checkpoints...")
    plot_loss_curves(history, output_dir / "gan_losses.png")

    model_path = output_dir / "fashion_generator.pt"
    torch.save(generator.state_dict(), model_path)
    print(f"      Saved generator checkpoint to: {model_path}")

    results_payload = {
        "device": str(device),
        "seed": RANDOM_SEED,
        "epochs": [
            {"epoch": i + 1, "D": round(d, 4), "G": round(g, 4)}
            for i, (d, g) in enumerate(history)
        ],
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("=" * 70)
    print("Execution finalized successfully. All lab outputs verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
