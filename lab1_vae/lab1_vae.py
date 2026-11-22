"""Laboratory Assignment 01: Variational Autoencoder (VAE) for Stochastic Latent Image Synthesis.

Course: ETMMGA304 - Generative AI and Its Applications
Institution: K.R. Mangalam University (Master of Computer Applications)
Curriculum Reference: BTX 3 (Apply) & BTX 4 (Analyze)

This script implements an end-to-end Variational Autoencoder (VAE) trained on the
MNIST handwritten digit benchmark. It demonstrates:
  1. Deep probabilistic encoder parameterized to output variational mean (mu) and log-variance.
  2. The Reparameterization Trick for backpropagation through stochastic nodes: z = mu + sigma * eps.
  3. Evidence Lower Bound (ELBO) optimization balancing Pixel Reconstruction and KL Divergence.
  4. Latent space traversal, empirical centroid clustering, and prior sampling from N(0, I).

Outputs:
  - lab_outputs/vae_mnist.pt: Serialized PyTorch state dictionary.
  - lab_outputs/vae_reconstructed.png: Paired original vs decoded reconstructions.
  - lab_outputs/vae_generated.png: 16 novel digits synthesized from standard normal noise.
  - lab_outputs/vae_latent.png: 2D manifold scatterplot of held-out test representations.
  - lab_outputs/results.json: Telemetry, training trajectories, and centroid distances.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid


# -----------------------------------------------------------------------------
# Configuration & Determinism
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.001
TOTAL_EPOCHS: int = 5
LATENT_DIMENSION: int = 2


def configure_environment(seed: int = RANDOM_SEED) -> Tuple[torch.device, Path]:
    """Establishes global random seed determinism and output directories."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("lab_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return device, output_dir


# -----------------------------------------------------------------------------
# Neural Architecture Definition
# -----------------------------------------------------------------------------
class VariationalAutoencoder(nn.Module):
    """Deep Variational Autoencoder with an explicit 2-dimensional latent bottleneck."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Probabilistic Inference Network (Encoder)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Generative Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input observations to variational parameters (mean and log-variance)."""
        features = self.encoder(x)
        mu = self.fc_mean(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick: z = mu + exp(0.5 * logvar) * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Projects latent vectors back into pixel observation space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through variational inference and reconstruction."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


# -----------------------------------------------------------------------------
# Loss Function: ELBO Decomposition
# -----------------------------------------------------------------------------
def compute_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the Negative Evidence Lower Bound (-ELBO).

    Loss = Reconstruction Error (BCE) + Kullback-Leibler Divergence (KL).
    """
    batch_size = x.size(0)
    # Sum over pixel dimensions to preserve proper relative weighting with KL
    reconstruction_loss = F.binary_cross_entropy(recon_x, x.flatten(1), reduction="sum")
    # Analytical KL divergence between q(z|x) ~ N(mu, sigma^2) and prior p(z) ~ N(0, I)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (reconstruction_loss + kl_divergence) / batch_size
    return total_loss, reconstruction_loss, kl_divergence


# -----------------------------------------------------------------------------
# Dataset Ingestion
# -----------------------------------------------------------------------------
def load_mnist_data(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, int, int]:
    """Loads and standardizes the MNIST dataset."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = str(project_root / "data")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


# -----------------------------------------------------------------------------
# Visualization & Artifact Generation
# -----------------------------------------------------------------------------
def save_image_grid(
    images: torch.Tensor,
    filename: Path,
    nrow: int = 4,
    figsize: Tuple[float, float] = (5.0, 5.0),
) -> None:
    """Renders and serializes an image tensor grid to disk."""
    grid = make_grid(images.cpu(), nrow=nrow)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()


def generate_latent_scatter(
    coords: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
) -> None:
    """Visualizes the 2D latent distribution colored by true digit ground truth."""
    plt.figure(figsize=(6.5, 4.5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Ground-Truth Digit Class", rotation=270, labelpad=15)
    plt.xlabel("Latent Bottleneck Dimension 1 ($z_1$)")
    plt.ylabel("Latent Bottleneck Dimension 2 ($z_2$)")
    plt.title("VAE 2D Latent Representation (Held-Out Test Set)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def compute_centroid_distances(
    coords: torch.Tensor,
    labels: torch.Tensor,
) -> List[Tuple[int, int, float]]:
    """Calculates pairwise Euclidean distances between latent digit class centroids."""
    centroids = {d: coords[labels == d].mean(dim=0) for d in range(10)}
    pairwise_distances = []
    for a in range(10):
        for b in range(a + 1, 10):
            dist = float(torch.norm(centroids[a] - centroids[b]))
            pairwise_distances.append((dist, a, b))

    pairwise_distances.sort(key=lambda item: item[0])
    return [(a, b, round(dist, 3)) for dist, a, b in pairwise_distances[:5]]


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    device, output_dir = configure_environment(RANDOM_SEED)
    print("=" * 70)
    print(f"ETMMGA304 Lab 01: Variational Autoencoder Execution | Device: {device}")
    print("=" * 70)

    # Step 1: Load Dataset
    print("[1/4] Ingesting MNIST benchmark...")
    train_loader, test_loader, num_train, num_test = load_mnist_data(BATCH_SIZE)

    # Step 2: Initialize Model & Optimizer
    print("[2/4] Constructing VAE architecture...")
    vae = VariationalAutoencoder(input_dim=784, hidden_dim=256, latent_dim=LATENT_DIMENSION).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    # Verification of shapes
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        rec_check, mu_check, logvar_check = vae(sample_batch)
    print(f"      Input Batch Shape: {tuple(sample_batch.shape)}")
    print(f"      Latent Mean/Logvar: {tuple(mu_check.shape)}")
    print(f"      Reconstruction Shape: {tuple(rec_check.shape)}")
    assert tuple(sample_batch.shape[1:]) == (1, 28, 28)
    assert mu_check.shape[1] == LATENT_DIMENSION and rec_check.shape[1] == 784

    # Step 3: Adversarial / Variational Optimization Loop
    print("\n[3/4] Commencing model training (5 epochs)...")
    epoch_logs: List[Dict[str, float]] = []

    for epoch in range(1, TOTAL_EPOCHS + 1):
        vae.train()
        accumulated_recon = 0.0
        accumulated_kl = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            reconstructed, mu, logvar = vae(images)
            loss, recon_term, kl_term = compute_vae_loss(reconstructed, images, mu, logvar)

            loss.backward()
            optimizer.step()

            accumulated_recon += recon_term.item()
            accumulated_kl += kl_term.item()

        avg_recon = round(accumulated_recon / num_train, 2)
        avg_kl = round(accumulated_kl / num_train, 2)
        total_loss = round(avg_recon + avg_kl, 2)

        assert math.isfinite(avg_recon) and math.isfinite(avg_kl), "Loss diverged to non-finite value!"
        log_entry = {
            "epoch": epoch,
            "reconstruction": avg_recon,
            "kl": avg_kl,
            "total": total_loss,
        }
        epoch_logs.append(log_entry)
        print(f"      Epoch [{epoch:02d}/{TOTAL_EPOCHS:02d}] -> Recon Loss: {avg_recon:.2f} | KL Div: {avg_kl:.2f} | Total: {total_loss:.2f}")

    # Save model weights
    checkpoint_path = output_dir / "vae_mnist.pt"
    torch.save(vae.state_dict(), checkpoint_path)
    print(f"\n      Saved trained checkpoint to: {checkpoint_path}")

    # Step 4: Sampling, Latent Inspection, and Held-Out Validation
    print("\n[4/4] Generating evaluation artifacts & held-out projections...")
    vae.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(test_loader))
        test_images = test_images.to(device)

        # 4a: Qualitative Reconstructions (Top 8 test images)
        mu_test, _ = vae.encode(test_images)
        reconstructions = vae.decoder(mu_test[:8]).reshape(-1, 1, 28, 28)
        comparison_tensor = torch.cat([test_images[:8].cpu(), reconstructions.cpu()])
        save_image_grid(comparison_tensor, output_dir / "vae_reconstructed.png", nrow=8, figsize=(8.0, 2.4))

        # 4b: Prior Sampling from N(0, I)
        standard_normal_noise = torch.randn(16, LATENT_DIMENSION, device=device)
        generated_samples = vae.decoder(standard_normal_noise).reshape(-1, 1, 28, 28)
        save_image_grid(generated_samples, output_dir / "vae_generated.png", nrow=4, figsize=(5.0, 5.0))

        # 4c: Latent Space Visualization
        coords = mu_test.cpu()
        generate_latent_scatter(coords, test_labels, output_dir / "vae_latent.png")

        # 4d: Full Held-Out Validation Loss
        val_recon = 0.0
        val_kl = 0.0
        for val_imgs, _ in test_loader:
            val_imgs = val_imgs.to(device)
            val_rec, v_mu, v_logvar = vae(val_imgs)
            val_recon += F.binary_cross_entropy(val_rec, val_imgs.flatten(1), reduction="sum").item()
            val_kl += (-0.5 * torch.sum(1 + v_logvar - v_mu.pow(2) - v_logvar.exp())).item()

        heldout_loss = {
            "reconstruction": round(val_recon / num_test, 2),
            "kl": round(val_kl / num_test, 2),
        }
        print(f"      Held-Out Evaluation -> Recon Loss: {heldout_loss['reconstruction']} | KL Div: {heldout_loss['kl']}")

        # 4e: Geometric Latent Overlap Analysis
        closest_pairs = compute_centroid_distances(coords, test_labels)
        print(f"      Closest Latent Centroid Pairs: {closest_pairs}")

    # Serialize evaluation results to JSON
    results_payload = {
        "device": str(device),
        "seed": RANDOM_SEED,
        "epochs": epoch_logs,
        "heldout_loss_per_image": heldout_loss,
        "closest_centroids": closest_pairs,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("=" * 70)
    print("Execution finalized successfully. All lab outputs verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
