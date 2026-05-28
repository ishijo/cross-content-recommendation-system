"""
Projection Head Training for Contrastive Learning

Trains a shared MLP projection head (768 → 128) using InfoNCE loss
to align book and movie embeddings in a lower-dimensional space.

Architecture:
    - Input: 768-dim embeddings (L2-normalized)
    - MLP: 768 → 512 → 256 → 128
    - Each layer: Linear → BatchNorm → ReLU → Dropout(0.2)
    - Output: 128-dim L2-normalized embeddings

Loss: InfoNCE with temperature=0.07
    - Positive pairs from genre overlap
    - In-batch negatives + 1 pre-mined hard negative per sample

Usage:
    python src/models/projection_head.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

from project_config import (
    EMBEDDINGS_DIR,
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    PROJECTION_HEAD_DIR
)
from utils.helpers import setup_logger, timer


class ProjectionHead(nn.Module):
    """
    MLP projection head: 768 → 512 → 256 → 128 with L2 normalization.

    Shared head for both book and movie embeddings.
    """

    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [512, 256],
                 output_dim: int = 128, dropout: float = 0.2):
        """
        Initialize projection head.

        Args:
            input_dim: Input embedding dimension (default 768)
            hidden_dims: Hidden layer dimensions (default [512, 256])
            output_dim: Output embedding dimension (default 128)
            dropout: Dropout probability (default 0.2)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # Batch normalization
            layers.append(nn.BatchNorm1d(dims[i+1]))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout))

        # Final projection layer (no activation or dropout after)
        layers.append(nn.Linear(dims[-1], output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.

        Args:
            x: Input embeddings (batch_size, 768)

        Returns:
            L2-normalized projected embeddings (batch_size, 128)
        """
        # Project
        projected = self.projection(x)

        # L2 normalize to unit sphere
        normalized = F.normalize(projected, p=2, dim=1)

        return normalized


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss with temperature scaling.

    Combines in-batch negatives with pre-mined hard negatives.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling (default 0.07)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, book_emb: torch.Tensor, movie_pos_emb: torch.Tensor,
                movie_neg_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            book_emb: Projected book embeddings (batch_size, 128)
            movie_pos_emb: Positive movie embeddings (batch_size, 128)
            movie_neg_emb: Hard negative movie embeddings (batch_size, 128)

        Returns:
            Scalar loss value
        """
        batch_size = book_emb.shape[0]

        # Positive similarity: dot product between book and its positive movie
        # Shape: (batch_size,)
        pos_sim = torch.sum(book_emb * movie_pos_emb, dim=1) / self.temperature

        # In-batch negatives: all other movies in the batch
        # Shape: (batch_size, batch_size)
        in_batch_sim = torch.matmul(book_emb, movie_pos_emb.T) / self.temperature

        # Hard negative similarity: pre-mined genre-incompatible movie
        # Shape: (batch_size,)
        hard_neg_sim = torch.sum(book_emb * movie_neg_emb, dim=1) / self.temperature

        # Construct logits matrix
        # Column 0: positive similarity
        # Columns 1 to batch_size: in-batch negative similarities
        # Column batch_size+1: hard negative similarity
        logits = torch.cat([
            pos_sim.unsqueeze(1),  # (batch_size, 1)
            in_batch_sim,  # (batch_size, batch_size)
            hard_neg_sim.unsqueeze(1)  # (batch_size, 1)
        ], dim=1)

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        # This is equivalent to: -log(exp(pos_sim) / sum(exp(all_sim)))
        loss = F.cross_entropy(logits, labels)

        return loss


class ContrastiveDataset(Dataset):
    """
    PyTorch dataset for contrastive learning.

    Yields triplets: (book_embedding, positive_movie_embedding, negative_movie_embedding)
    """

    def __init__(self, pairs_path: Path, embeddings_dir: Path):
        """
        Initialize dataset.

        Args:
            pairs_path: Path to contrastive_pairs.json
            embeddings_dir: Path to embeddings directory
        """
        # Load pairs
        with open(pairs_path) as f:
            data = json.load(f)

        self.positive_pairs = data['positive_pairs']
        self.hard_negatives = data['hard_negatives']

        # Load embeddings into memory
        self.book_embeddings = np.load(embeddings_dir / "book_embeddings.npy")
        self.movie_embeddings = np.load(embeddings_dir / "movie_embeddings.npy")

        # Convert to float32 for torch
        self.book_embeddings = self.book_embeddings.astype('float32')
        self.movie_embeddings = self.movie_embeddings.astype('float32')

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Index

        Returns:
            (book_embedding, positive_movie_embedding, negative_movie_embedding)
        """
        # Get positive pair
        book_idx, movie_pos_idx = self.positive_pairs[idx]

        # Get hard negative
        key = f"{book_idx}_{movie_pos_idx}"
        movie_neg_idx = self.hard_negatives[key]

        # Get embeddings
        book_emb = torch.from_numpy(self.book_embeddings[book_idx])
        movie_pos_emb = torch.from_numpy(self.movie_embeddings[movie_pos_idx])
        movie_neg_emb = torch.from_numpy(self.movie_embeddings[movie_neg_idx])

        return book_emb, movie_pos_emb, movie_neg_emb


class ContrastiveTrainer:
    """Handles training loop, validation, and checkpointing"""

    def __init__(self, config: Dict, logger):
        """
        Initialize trainer.

        Args:
            config: Training configuration dict
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Model
        self.model = ProjectionHead(
            input_dim=768,
            hidden_dims=[512, 256],
            output_dim=128,
            dropout=config['dropout']
        ).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {n_params:,}")

        # Loss
        self.criterion = InfoNCELoss(temperature=config['temperature'])

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # LR Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for book_emb, movie_pos_emb, movie_neg_emb in pbar:
            # Move to device
            book_emb = book_emb.to(self.device)
            movie_pos_emb = movie_pos_emb.to(self.device)
            movie_neg_emb = movie_neg_emb.to(self.device)

            # Project embeddings
            book_proj = self.model(book_emb)
            movie_pos_proj = self.model(movie_pos_emb)
            movie_neg_proj = self.model(movie_neg_emb)

            # Compute loss
            loss = self.criterion(book_proj, movie_pos_proj, movie_neg_proj)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / n_batches
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for book_emb, movie_pos_emb, movie_neg_emb in val_loader:
                # Move to device
                book_emb = book_emb.to(self.device)
                movie_pos_emb = movie_pos_emb.to(self.device)
                movie_neg_emb = movie_neg_emb.to(self.device)

                # Project
                book_proj = self.model(book_emb)
                movie_pos_proj = self.model(movie_pos_emb)
                movie_neg_proj = self.model(movie_neg_emb)

                # Loss
                loss = self.criterion(book_proj, movie_pos_proj, movie_neg_proj)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        if is_best:
            path = PROJECTION_HEAD_DIR / "best_model.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history dict
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STARTING TRAINING")
        self.logger.info("="*70)

        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Track
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Log
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"lr={current_lr:.6f}"
            )

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config['patience']:
                self.logger.info(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break

        self.logger.info("\n✓ Training complete!")
        self.logger.info(f"  Best validation loss: {self.best_val_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


def compute_similarity_gap(model: ProjectionHead, dataset: ContrastiveDataset,
                          n_samples: int = 1000, device: torch.device = None) -> Dict:
    """
    Compute similarity gap improvement: before vs after projection.

    Args:
        model: Trained projection head
        dataset: Contrastive dataset
        n_samples: Number of samples to evaluate
        device: Torch device

    Returns:
        Dict with before/after similarity statistics
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()

    # Sample random pairs
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    before_sims = []
    after_sims = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing similarity gap"):
            book_emb, movie_pos_emb, _ = dataset[idx]

            # Before: raw 768-dim similarity
            before_sim = float(torch.dot(book_emb, movie_pos_emb))
            before_sims.append(before_sim)

            # After: projected 128-dim similarity
            book_emb = book_emb.unsqueeze(0).to(device)
            movie_pos_emb = movie_pos_emb.unsqueeze(0).to(device)

            book_proj = model(book_emb)
            movie_proj = model(movie_pos_emb)

            after_sim = float(torch.dot(book_proj.squeeze(), movie_proj.squeeze()))
            after_sims.append(after_sim)

    before_mean = float(np.mean(before_sims))
    after_mean = float(np.mean(after_sims))
    improvement = after_mean - before_mean
    improvement_pct = (improvement / abs(before_mean)) * 100

    return {
        'before_mean': before_mean,
        'before_std': float(np.std(before_sims)),
        'after_mean': after_mean,
        'after_std': float(np.std(after_sims)),
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }


def plot_training_curves(history: Dict, output_path: Path):
    """
    Plot training and validation loss curves.

    Args:
        history: Training history dict
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_losses']) + 1)

    ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)

    # Mark best validation loss
    best_epoch = np.argmin(history['val_losses']) + 1
    best_val_loss = min(history['val_losses'])
    ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_val_loss], color='r', s=100, zorder=5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Contrastive Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training pipeline"""
    logger = setup_logger("contrastive_trainer")

    logger.info("\n" + "="*70)
    logger.info("CONTRASTIVE PROJECTION HEAD TRAINING")
    logger.info("="*70)

    # Configuration
    config = {
        'lr': 1e-4,
        'batch_size': 256,
        'epochs': 50,
        'patience': 5,
        'temperature': 0.07,
        'dropout': 0.2,
        'train_split': 0.8,
        'weight_decay': 1e-5,
        'seed': 42
    }

    logger.info("\n📋 Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load dataset
    logger.info("\n" + "="*70)
    logger.info("LOADING DATASET")
    logger.info("="*70)

    with timer("Loading contrastive pairs", logger):
        pairs_path = PROCESSED_DATA_DIR / "contrastive_pairs.json"
        dataset = ContrastiveDataset(pairs_path, EMBEDDINGS_DIR)
        logger.info(f"  Total samples: {len(dataset):,}")

    # Train/val split
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Val samples: {len(val_dataset):,}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"  Train batches: {len(train_loader):,}")
    logger.info(f"  Val batches: {len(val_loader):,}")

    # Train
    trainer = ContrastiveTrainer(config, logger)
    history = trainer.train(train_loader, val_loader)

    # Compute similarity gap
    logger.info("\n" + "="*70)
    logger.info("COMPUTING SIMILARITY GAP")
    logger.info("="*70)

    gap_stats = compute_similarity_gap(
        trainer.model,
        dataset,
        n_samples=1000,
        device=trainer.device
    )

    logger.info(f"\n📊 Similarity Gap Analysis:")
    logger.info(f"  Before projection: {gap_stats['before_mean']:.4f} ± {gap_stats['before_std']:.4f}")
    logger.info(f"  After projection:  {gap_stats['after_mean']:.4f} ± {gap_stats['after_std']:.4f}")
    logger.info(f"  Improvement: {gap_stats['improvement']:+.4f} ({gap_stats['improvement_pct']:+.1f}%)")

    # Plot training curves
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)

    with timer("Plotting training curves", logger):
        plot_path = PLOTS_DIR / "contrastive_training_curves.png"
        plot_training_curves(history, plot_path)
        logger.info(f"  ✓ Saved plot: {plot_path}")

    # Save training config
    config_path = PROJECTION_HEAD_DIR / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            **config,
            'best_val_loss': history['best_val_loss'],
            'n_epochs_trained': len(history['train_losses']),
            'similarity_gap': gap_stats
        }, f, indent=2)
    logger.info(f"  ✓ Saved config: {config_path}")

    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nModel checkpoint: {PROJECTION_HEAD_DIR / 'best_model.pt'}")
    logger.info(f"Best val loss: {history['best_val_loss']:.4f}")
    logger.info(f"Similarity improvement: {gap_stats['improvement_pct']:+.1f}%")


if __name__ == "__main__":
    main()
