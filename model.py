import os
import warnings
from collections import OrderedDict
from typing import Iterable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from utils import SoftClusterAssignment

warnings.filterwarnings("ignore")


class FeatureDataset(Dataset):
    """Feature dataset for storing encoder-extracted features during layer-wise pretraining."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class IndexedDataset(Dataset):
    """Dataset wrapper with global indices, used in DEC training to match pre-computed target distribution P.

    When DataLoader uses shuffle=True, global indices are needed to align with the pre-computed target distribution P.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return idx, data, label


def cluster_acc(y_true, y_pred):
    """Compute clustering accuracy using the Hungarian algorithm.

    Algorithm:
    1. Build a confusion matrix w where w[i,j] is the count of samples predicted as i with true label j
    2. Use the Hungarian algorithm (linear_sum_assignment) to find the optimal label mapping
    3. Compute the ratio of correctly matched samples

    Args:
        y_true: Ground truth labels, numpy array of shape (n_samples,)
        y_pred: Predicted labels, numpy array of shape (n_samples,)

    Returns:
        Accuracy in range [0, 1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)

    accuracy = 0
    for i, j in zip(ind[0], ind[1]):
        accuracy += w[i, j]
    accuracy = accuracy * 1.0 / y_pred.size
    return accuracy


class AutoEncoder(pl.LightningModule):
    """Single autoencoder for layer-wise pretraining.

    Structure: Input -> Hidden -> Reconstruction

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        activation: Activation function module
        dropout: Dropout rate
        batch_size: Batch size
        lr: Learning rate
        lr_decay: Learning rate decay factor
        lr_decay_step: Learning rate decay step (iterations)
        drop_last: Whether to drop the last incomplete batch
        custom_dset: Custom dataset (for layer-wise pretraining with feature datasets)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: Optional[float] = 0.2,
        batch_size: int = 256,
        lr: float = 0.1,
        lr_decay: float = 0.1,
        lr_decay_step: int = 20000,
        drop_last: bool = False,
        opt: str = 'SGD',
        custom_dset: Optional[torch.utils.data.Dataset] = None,
    ):
        super(AutoEncoder, self).__init__()

        self.criterion = nn.MSELoss()
        self.opt = opt
        # Encoder: Input -> Hidden
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        if activation is not None:
            encoder_layers.append(activation)
        if dropout is not None:
            encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: Hidden -> Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )
        if input_dim == 784:
            self.decoder.append(nn.Sigmoid())

        self.encoder.apply(self._init_weight)
        self.decoder.apply(self._init_weight)

        self.save_hyperparameters(ignore=['activation', 'custom_dset', 'opt'])

        self.custom_dset = custom_dset

        print(f"AutoEncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")
        print(f"  Encoder first layer: Linear({input_dim}, {hidden_dim})")
        print(f"  Decoder first layer: Linear({hidden_dim}, {input_dim})")

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def _init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def prepare_data(self) -> None:
        """Prepare data: use custom dataset if provided, otherwise load MNIST.

        Data preprocessing matches the original Keras implementation:
        - ToTensor: converts PIL images to tensors, pixel values normalized to [0,1]
        This is equivalent to Keras' np.divide(x, 255.)
        """
        if self.custom_dset is not None:
            self.dset = self.custom_dset
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
            test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
            self.dset = ConcatDataset([train_dset, test_dset])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=self.hparams.drop_last,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def configure_optimizers(self):
        if self.opt == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.opt == 'Adam':
            if self.hparams.lr > 0.001:
                self.hparams.lr = 0.001
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        # Reduce learning rate by half when loss plateaus for 5 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            }
        }

    def training_step(self, batch, batch_idx) -> dict:
        data, _ = batch
        flatten = data.reshape(data.size(0), -1)
        reconstruction = self(flatten)
        loss = self.criterion(reconstruction, flatten)
        self.log("train_loss", loss)
        Vlr = self.optimizers().param_groups[0]['lr']
        self.log("lr", Vlr, on_step=True, on_epoch=False)
        return {"loss": loss}


class SAE(pl.LightningModule):
    """Stacked AutoEncoder (SAE), supporting layer-wise pretraining and end-to-end fine-tuning.

    Model structure:
    Encoder: Input -> Hidden1 -> Hidden2 -> ... -> Code layer (low-dimensional features)
    Decoder: Code layer -> Hidden2 -> Hidden1 -> ... -> Output (reconstruction)

    Layer-wise pretraining flow:
    1. Train AE1: Input(784) -> Hidden1(500) -> Reconstruction(784)
    2. Train AE2: Hidden1 output(500) -> Hidden2(500) -> Reconstruction(500)
    3. Train AE3: Hidden2 output(500) -> Hidden3(2000) -> Reconstruction(500)
    4. Train AE4: Hidden3 output(2000) -> Code(10) -> Reconstruction(2000)
    5. Stack all layers to form the complete SAE
    6. End-to-end fine-tuning of the entire network

    Args:
        dimensions: Network dimensions, format [input_dim, hidden_dim_1, ..., hidden_dim_n]
        activation: Non-linear activation function for encoder and decoder
        final_activation: Non-linear activation function for the final layer
        dropout: Dropout rate for each layer
        batch_size: Mini-batch size
        lr: Learning rate
        lr_decay: Learning rate decay factor
        lr_decay_step: Learning rate decay step (iterations)
        weight_decay: Weight decay coefficient
    """

    def __init__(
        self,
        dimensions: Iterable[int],
        activation: Optional[nn.Module] = nn.ReLU(),
        final_activation: Optional[nn.Module] = nn.Sigmoid(),
        dropout: Optional[float] = 0.0,
        batch_size: int = 256,
        lr: float = 0.1,
        lr_decay: float = 0.1,
        lr_decay_step: int = 20000,
        drop_last: bool = False,
        opt: str = 'SGD',
        weight_decay: float = 0.0,
    ):
        super(SAE, self).__init__()

        self.criterion = nn.MSELoss()
        self.opt = opt
        # Build encoder layers: from input to hidden layers, last layer without activation and dropout
        encoder_layers = self._add_linear_layer_stack(
            dimensions[:-1], activation, dropout
        )
        # Add final layer: from second-to-last dimension to code dimension, no activation or dropout
        encoder_layers.extend(
            self._add_linear_layer_stack(
                [dimensions[-2], dimensions[-1]], None, dropout=None
            )
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers: dimensions reversed from encoder
        decoder_layers = self._add_linear_layer_stack(
            list(reversed(dimensions[1:])), activation, dropout=None
        )
        # Add final layer: from first hidden dimension to output dimension
        decoder_layers.extend(
            self._add_linear_layer_stack(
                [dimensions[1], dimensions[0]], final_activation, dropout=None
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        self.encoder.apply(self._init_weight)
        self.decoder.apply(self._init_weight)

        self.save_hyperparameters(ignore=['activation', 'final_activation'])

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode, return reconstruction."""
        encoded = self.encoder(batch)
        return self.decoder(encoded)

    def prepare_data(self) -> None:
        """Prepare data: load MNIST and preprocess.

        Data preprocessing matches the original Keras implementation:
        - ToTensor: converts PIL images to tensors, pixel values normalized to [0,1]
        This is equivalent to Keras' np.divide(x, 255.)

        In unsupervised clustering, there is no validation set for cross-validation,
        so we merge training and test sets.
        """
        transform = transforms.Compose([transforms.ToTensor()])
        train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
        self.dset = ConcatDataset([train_dset, test_dset])

    def _add_linear_layer_stack(
        self,
        dims: Iterable[int],
        activation: Optional[nn.Module],
        dropout: Optional[float],
    ) -> List[nn.Module]:
        """Build a stack of linear layers, each with optional activation and dropout.

        Args:
            dims: List of dimensions, e.g. [784, 500, 500]
            activation: Activation function, e.g. ReLU()
            dropout: Dropout rate. If None, no Dropout layer is added.

        Returns:
            List of Sequential modules, each containing a Linear layer, activation, and dropout.
        """
        def single_unit(in_dim: int, out_dim: int) -> List[nn.Module]:
            unit = [nn.Linear(in_dim, out_dim)]
            if activation is not None:
                unit.append(activation)
            if dropout is not None:
                unit.append(nn.Dropout(dropout))
            return nn.Sequential(*unit)

        return [single_unit(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]

    def _init_weight(self, layer):
        """Initialize weights with uniform distribution: [-sqrt(1./fan_in), sqrt(1./fan_in)]"""
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=self.hparams.drop_last,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def configure_optimizers(self):
        if self.opt == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.opt == 'Adam':
            if self.hparams.lr > 0.001:
                self.hparams.lr = 0.001
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            }
        }

    def training_step(self, batch, batch_idx) -> dict:
        data, _ = batch
        flatten = data.reshape(data.size(0), -1)
        reconstruction = self(flatten)
        loss = self.criterion(reconstruction, flatten)

        self.log("train_loss", loss)
        Vlr = self.optimizers().param_groups[0]['lr']
        self.log("lr", Vlr, on_step=True, on_epoch=False)
        return {"loss": loss}

    def set_dropout(self, rate: float):
        """Modify the dropout probability of all Dropout layers in the model."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def save(self, path: str) -> None:
        """Save SAE model weights to the specified path.

        Args:
            path: Save path, can be a .ckpt file or directory
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
        }, path)
        print(f"SAE model saved to: {path}")

    def visualize_reconstruction(self, num_samples=16, device=None):
        """Visualize the reconstruction quality of the autoencoder.

        Args:
            num_samples: Number of samples to display (recommended 8~16)
            device: Device to use; auto-inferred if None
        """
        if not hasattr(self, 'dset') or self.dset is None:
            self.prepare_data()

        if device is None:
            device = next(self.parameters()).device

        loader = DataLoader(self.dset, batch_size=num_samples, shuffle=True)
        data, _ = next(iter(loader))
        data = data.to(device).view(num_samples, -1)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            recon = self(data)

        if was_training:
            self.train()

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 1.5, 3))
        for i in range(num_samples):
            axes[0, i].imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon[i].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.show()

    @classmethod
    def load_from_checkpoint(cls, path: str, **kwargs) -> 'SAE':
        """Load an SAE model from a saved checkpoint file.

        Args:
            path: Checkpoint file path (.ckpt)
            **kwargs: Additional initialization parameters that override saved hparams

        Returns:
            SAE instance with loaded weights
        """
        checkpoint = torch.load(path, map_location='cpu')
        hparams = checkpoint['hparams'].copy()
        strict = kwargs.pop('strict', True)
        hparams.update(kwargs)

        if 'opt' not in hparams:
            hparams['opt'] = 'SGD'
            print("Warning: 'opt' not found in checkpoint, defaulting to 'SGD'")

        model = cls(**hparams)
        state_dict = checkpoint['state_dict']

        # Key remapping: flat -> nested
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                parts = key.split('.')
                if len(parts) == 2:
                    idx = int(parts[1])
                    if idx in [0, 3, 6, 9]:
                        new_idx = idx // 3
                        name = parts[0]
                        new_key = f'encoder.{new_idx}.0.{name}'
                        new_state_dict[new_key] = value
                    else:
                        continue
                else:
                    new_state_dict[key] = value
            elif key.startswith('decoder.'):
                parts = key.split('.')
                if len(parts) == 2:
                    idx = int(parts[1])
                    if idx in [0, 3, 6, 9]:
                        new_idx = idx // 3
                        name = parts[0]
                        new_key = f'decoder.{new_idx}.0.{name}'
                        new_state_dict[new_key] = value
                    else:
                        continue
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value

        missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        print(f"SAE model loaded from: {path}")
        return model

    @classmethod
    def build_from_layerwise_pretraining(
        cls,
        dimensions: List[int],
        activation: nn.Module = nn.ReLU(),
        final_activation: nn.Module = nn.Sigmoid(),
        dropout: float = 0.2,
        batch_size: int = 256,
        lr: float = 0.1,
        lr_decay: float = 0.1,
        lr_decay_step: int = 20000,
        opt: str = 'SGD',
        drop_last: bool = False,
        epoch_pretrain: int = 300,
        patience: int = 20,
        trainer_kwargs: dict = None,
    ):
        """Build a stacked autoencoder through layer-wise pretraining.

        Layer-wise pretraining flow:
        1. Train AE1: Input(784) -> Hidden1(500) -> Reconstruction(784)
        2. Train AE2: AE1 encoder output(500) -> Hidden2(500) -> Reconstruction(500)
        3. Train AE3: AE1+AE2 encoder output(500) -> Hidden3(2000) -> Reconstruction(500)
        4. Train AE4: AE1+AE2+AE3 encoder output(2000) -> Code(10) -> Reconstruction(2000)
        5. Stack all layers to form the complete SAE

        Args:
            dimensions: Network dimensions, e.g. [784, 500, 500, 2000, 10]
            activation: Activation function
            dropout: Dropout rate
            batch_size: Batch size
            lr: Learning rate
            lr_decay: Learning rate decay factor
            lr_decay_step: Learning rate decay step
            drop_last: Whether to drop the last incomplete batch
            trainer_kwargs: Additional Trainer arguments

        Returns:
            SAE: Stacked autoencoder with layer-wise pretrained weights
        """
        if trainer_kwargs is None:
            trainer_kwargs = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoder_layers = []
        decoder_layers = []

        # Load raw data
        transform = transforms.Compose([transforms.ToTensor()])
        train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
        full_dset = ConcatDataset([train_dset, test_dset])

        current_encoder = None

        # Layer-wise pretraining
        for i in range(len(dimensions) - 1):
            input_dim = dimensions[i]
            hidden_dim = dimensions[i + 1]

            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"#     Layer-wise pretrain layer {i+1}: {input_dim} → {hidden_dim}     #")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            dloader = DataLoader(
                full_dset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

            if current_encoder is not None:
                print(f"  Extracting features with previous {i} encoder layers...")
                current_encoder.eval()
                current_encoder = current_encoder.to(device)

                features = []
                labels = []
                with torch.no_grad():
                    for batch in dloader:
                        data, target = batch
                        data = data.to(device)
                        flatten = data.reshape(data.size(0), -1)
                        feat = current_encoder(flatten)
                        features.append(feat.cpu())
                        labels.append(target)

                features = torch.cat(features)
                labels = torch.cat(labels)

                lr = 0.1
                feat_dset = FeatureDataset(features, labels)
                print(f"  Feature extraction complete, {len(feat_dset)} samples total")
            else:
                feat_dset = full_dset

            ae = AutoEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                dropout=dropout,
                batch_size=batch_size,
                lr=lr,
                lr_decay=lr_decay,
                lr_decay_step=lr_decay_step,
                opt=opt,
                drop_last=drop_last,
                custom_dset=feat_dset,
            )

            trainer = pl.Trainer(
                max_epochs=epoch_pretrain,
                callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=patience)],
                deterministic=True,
                logger=CSVLogger("logs", name=f"layerwise_ae{i+1}"),
                enable_checkpointing=False,
                **trainer_kwargs,
            )
            trainer.fit(ae)

            encoder_layer = nn.Linear(input_dim, hidden_dim)
            encoder_layer.load_state_dict(list(ae.encoder.children())[0].state_dict())

            encoder_layers.append(encoder_layer)

            if i < len(dimensions) - 2:
                encoder_layers.append(type(activation)())
                if dropout is not None:
                    encoder_layers.append(nn.Dropout(dropout))

            decoder_layer = nn.Linear(hidden_dim, input_dim)
            decoder_layer.load_state_dict(list(ae.decoder.children())[0].state_dict())

            decoder_layers.append(decoder_layer)

            current_encoder = nn.Sequential(*encoder_layers)

        # Build the full decoder (reverse order and add activations)
        full_decoder_layers = []
        for j, dim in enumerate(reversed(dimensions[1:])):
            layer = decoder_layers[len(decoder_layers) - 1 - j]
            full_decoder_layers.append(layer)
            if j < len(dimensions) - 2:
                full_decoder_layers.append(activation)
                if dropout is not None:
                    full_decoder_layers.append(nn.Dropout(dropout))
        full_decoder_layers.append(final_activation)

        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*full_decoder_layers)

        lr = 1.0
        sae = cls(
            dimensions=dimensions,
            activation=activation,
            dropout=dropout,
            batch_size=batch_size,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_step=lr_decay_step,
            drop_last=drop_last,
        )
        for name, param in sae.named_parameters():
            if torch.isnan(param).any():
                print(f"Parameter {name} contains NaN, please retrain!")
                raise RuntimeError("Pretrained weights are corrupted")

        # Map flat weights to nested structure and load
        flat_state_dict = {}
        flat_state_dict.update(encoder.state_dict())
        flat_state_dict.update(decoder.state_dict())

        mapped_state_dict = {}
        for key, value in flat_state_dict.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx % 3 == 0:
                    new_idx = idx // 3
                    new_key = f"{parts[0]}.{new_idx}.0.weight" if 'weight' in key else f"{parts[0]}.{new_idx}.0.bias"
                    mapped_state_dict[new_key] = value
            else:
                mapped_state_dict[key] = value

        sae.load_state_dict(mapped_state_dict, strict=True)

        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"#     Layer-wise pretraining complete, starting end-to-end fine-tuning     #")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return sae


class DEC(pl.LightningModule):
    """Deep Embedded Clustering (DEC).

    DEC is an unsupervised clustering method that learns low-dimensional embeddings
    while optimizing a clustering objective.

    Core idea:
    1. Use a pre-trained autoencoder as the feature extractor
    2. Initialize cluster centers with K-Means
    3. Iteratively optimize the KL divergence loss, updating both network parameters and cluster centers

    Loss function: KL(q || p), where
    - q: Soft cluster assignment based on Student's t-distribution
    - p: Target distribution, obtained by sharpening high-confidence assignments in q

    Args:
        encoder: Fine-tuned stacked autoencoder encoder
        num_cluster: Number of clusters
        hidden_dim: Dimension of the encoder output vector
        alpha: Degrees of freedom parameter of the t-distribution
        batch_size: Batch size
        lr_dec: Learning rate for DEC training
        tol: Tolerance for stopping criterion
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_cluster: int = 10,
        hidden_dim: int = 10,
        alpha: float = 1.0,
        batch_size: int = 256,
        lr_dec: float = 0.01,
        tol: float = 1e-3,
        drop_last: bool = False,
        update_interval: int = 140,
        maxiter: int = 20000,
    ):
        super(DEC, self).__init__()

        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder

        self.assignment = SoftClusterAssignment(num_cluster, hidden_dim, alpha)

        self.kmeans = KMeans(self.hparams.num_cluster, n_init=20)

        self.init = True

        self.y_pred_last = None

        self.global_p = None

        self.global_q = None

    def forward(self, batch):
        """Forward pass: encode input and compute soft cluster assignment.

        Returns:
            q: Soft assignment matrix, shape (batch_size, num_cluster)
        """
        encoded = self.encoder(batch)
        return self.assignment(encoded)

    def prepare_data(self) -> None:
        """Prepare data: load MNIST and preprocess.

        Data preprocessing matches the original Keras implementation:
        - ToTensor: converts PIL images to tensors, pixel values normalized to [0,1]
        This is equivalent to Keras' np.divide(x, 255.)

        In unsupervised clustering, there is no validation set for cross-validation,
        so we merge training and test sets.
        """
        transform = transforms.Compose([transforms.ToTensor()])

        train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
        self.dset = ConcatDataset([train_dset, test_dset])

        # Store true labels for evaluation
        y_train = train_dset.targets.numpy()
        y_test = test_dset.targets.numpy()
        self.y_true = np.concatenate([y_train, y_test])

        # Wrap with global index dataset for matching pre-computed target distribution P
        self.indexed_dset = IndexedDataset(self.dset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.indexed_dset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=self.hparams.drop_last,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader (no shuffle).
        Validation is used to compute clustering accuracy, no need to shuffle data.
        """
        return DataLoader(
            self.dset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def configure_optimizers(self) -> torch.optim:
        """Configure optimizer (SGD with small learning rate).
        DEC uses a small learning rate (0.01) to fine-tune the pre-trained encoder
        without destroying learned feature representations.
        """
        return optim.SGD(self.parameters(), lr=self.hparams.lr_dec, momentum=0.9)

    def training_step(self, batch, batch_idx) -> dict:
        """Training step:
        1. First iteration: initialize cluster centers with KMeans
        2. Every update_interval iterations: recompute global target distribution P and check convergence
        3. Use global P to compute KL divergence loss

        Soft assignment q (t-distribution):
        q_ij = (1 + ||z_i - c_j||^2 / alpha)^(-(alpha+1)/2) / sum_j(...)

        Target distribution p (based on full data):
        p_ij = q_ij^2 / n_j / sum_i(q_ij^2 / n_j)
        where n_j = sum_i(q_ij) is the global weight of cluster j

        Loss: KL(q || p) = sum_i sum_j q_ij * log(q_ij / p_ij)
        """
        if self.global_step >= self.hparams.maxiter:
            self.trainer.should_stop = True
            return {"loss": torch.tensor(0.0)}

        if self.init:
            init_info = self._initialize_centroid()
            self.assignment = SoftClusterAssignment(
                self.hparams.num_cluster,
                self.hparams.hidden_dim,
                self.hparams.alpha,
                init_info["centroid"],
            )

            print(f"Initial accuracy: {init_info['accuracy']}")
            self.init = False

        if self.global_step % self.hparams.update_interval == 0:
            self._compute_global_q_and_p()

        indices, data, target = batch

        q = self(data.reshape(data.size(0), -1))

        p = self.global_p[indices.cpu()].to(self.device).detach()

        loss = F.kl_div(q.log(), p, reduction='batchmean')
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> dict:
        """Validation step: compute clustering accuracy."""
        data, target = batch
        embedded = self(data.reshape(data.size(0), -1))
        pred = embedded.max(1)[1]

        accuracy = cluster_acc(target.cpu().numpy(), pred.cpu().numpy())
        self.log("accuracy", accuracy)
        return {"accuracy": accuracy}

    def _initialize_centroid(self) -> dict:
        """Initialize cluster centers using KMeans.

        Process:
        1. Iterate over all data and extract features using the encoder
        2. Apply KMeans clustering on features (n_init=20, take best result)
        3. Compute initial clustering accuracy (using Hungarian algorithm)

        Returns:
            dict: Contains "accuracy" (initial accuracy) and "centroid" (cluster center tensor)
        """
        print("Initializing cluster centers...")
        for name, param in self.encoder.named_parameters():
            if torch.isnan(param).any():
                print(f"Encoder parameter {name} contains NaN!")
        dloader = DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=False
        )
        label, feature = [], []

        for batch in dloader:
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            label.append(target)
            feature.append(
                self.encoder(data.reshape(data.size(0), -1)).detach().cpu()
            )

        label = torch.cat(label)
        pred = self.kmeans.fit_predict(torch.cat(feature).numpy())
        accuracy = cluster_acc(label.cpu().numpy(), pred)

        return {
            "accuracy": accuracy,
            "centroid": torch.tensor(
                self.kmeans.cluster_centers_,
                requires_grad=True,
                device=self.device,
            ),
        }

    def _get_target_distribution(self, q):
        """Compute target distribution p.

        Formula:
        p_ij = q_ij^2 / n_j / sum_i(q_ij^2 / n_j)

        where n_j = sum_i(q_ij) is the cluster j's frequency.

        Design principle:
        Squaring q_ij amplifies high-confidence assignments and suppresses low-confidence ones,
        making the target distribution p more "sharp" and guiding the model toward clearer cluster boundaries.

        Args:
            q: Soft assignment matrix, shape (batch_size, num_cluster)

        Returns:
            p: Target distribution matrix, same shape as q
        """
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 1)).t()
        return p

    def _compute_global_q_and_p(self):
        """Compute global soft assignment Q and target distribution P over the full dataset,
        and check for convergence.

        DEC paper's core flow:
        1. Forward pass over all data to get soft assignment Q for each sample
        2. Compute target distribution P from global Q
        3. Store P as a global variable for subsequent training batches
        4. Compute delta_label to check convergence; stop if delta_label < tol

        Target distribution P is recomputed only every update_interval iterations,
        providing a stable optimization target for the network.
        """
        if not hasattr(self, '_eval_loader'):
            self._eval_loader = DataLoader(
                self.dset,
                batch_size=self.hparams.batch_size * 4,
                shuffle=False,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            )

        q_list = []
        self.encoder.eval()
        self.assignment.eval()
        with torch.no_grad():
            for batch in self._eval_loader:
                data, _ = batch
                data = data.to(self.device)
                q = self(data.reshape(data.size(0), -1)).cpu()
                q_list.append(q)

        self.global_q = torch.cat(q_list)
        self.global_p = self._get_target_distribution(self.global_q)

        y_pred = self.global_q.max(1)[1].numpy()

        if self.y_pred_last is not None:
            delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            print(f"Iter {self.global_step}: delta_label = {delta_label:.6f}, tol = {self.hparams.tol}")
            if delta_label < self.hparams.tol:
                print(f"Iter {self.global_step}: Convergence threshold reached, stopping training")
                self.trainer.should_stop = True

        self.y_pred_last = y_pred
        self.encoder.train()
        self.assignment.train()

    def save(self, path: str) -> None:
        """Save DEC model weights to the specified path.

        Saved content:
        - Encoder weights (from SAE)
        - Cluster assignment layer weights (cluster centers)
        - Hyperparameters

        Args:
            path: Save path, can be a .ckpt file or directory
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'assignment_state_dict': self.assignment.state_dict(),
            'hparams': self.hparams,
        }, path)
        print(f"DEC model saved to: {path}")

    def evaluate(self, dataloader=None, y_true=None):
        """Evaluate clustering performance, returning ACC, NMI, ARI.

        Args:
            dataloader: Optional; if not provided, uses self.val_dataloader()
            y_true: Optional; external ground truth labels. If None, tries self.y_true,
                    otherwise collects from dataloader.

        Returns:
            dict: {'ACC': acc, 'NMI': nmi, 'ARI': ari}
        """
        if dataloader is None:
            dataloader = self.val_dataloader()

        self.eval()
        all_pred = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    data, target = batch
                elif len(batch) == 3:
                    _, data, target = batch
                else:
                    raise ValueError("Unexpected batch format")
                data = data.view(data.size(0), -1).to(self.device)
                q = self(data)
                pred = q.argmax(dim=1).cpu().numpy()
                all_pred.extend(pred)
                all_labels.extend(target.numpy())

        y_pred = np.array(all_pred)

        if y_true is not None:
            y_true = np.array(y_true)
        elif hasattr(self, 'y_true') and self.y_true is not None:
            y_true = self.y_true
        else:
            y_true = np.array(all_labels)

        assert len(y_true) == len(y_pred), "Length mismatch"

        acc = cluster_acc(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)

        return {'ACC': acc, 'NMI': nmi, 'ARI': ari}