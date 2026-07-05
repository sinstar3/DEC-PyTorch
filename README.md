# DEC-PyTorch (Improved) —— 87.38% ACC on MNIST

> PyTorch Lightning implementation of **"Unsupervised Deep Embedding for Clustering Analysis"** (ICML 2016).  
> This repository fixes several critical issues and optimizes the implementation based on [youngerous/dec-pytorch](https://github.com/youngerous/dec-pytorch), achieving **87.38% clustering accuracy on MNIST** (surpassing the original paper's reported 84%).

## 🎯 Key Improvements

Compared to the original reference code, this version includes the following critical fixes and optimizations:

- **Bottleneck Activation**: Changed from `Sigmoid` to **linear (no activation)**, preventing feature space compression and enabling K-Means to better distinguish clusters.
- **Weight Initialization**: Replaced with **Kaiming Uniform** (`mode="fan_in", nonlinearity='relu'`), perfectly suited for ReLU and effectively alleviating gradient vanishing.
- **Target Distribution P Calculation**: Changed from mini-batch based to **full dataset based**, updated every `update_interval` steps to ensure global stability of clustering objectives.
- **Learning Rate Strategy**: Independent learning rates for different training phases (layer-wise pretraining, end-to-end fine-tuning, DEC clustering) to avoid gradient explosion.
- **Gradient Clipping**: Enabled gradient clipping (`gradient_clip_val=0.5`) during fine-tuning and DEC training for improved stability.

## 📊 Experimental Results (MNIST)

| Metric | This Repository |
|--------|-----------------|
| **ACC** | **87.38%** |
| **NMI** | 0.8433 |
| **ARI** | 0.8085 |

> The original DEC paper reported ~84% ACC; this implementation significantly outperforms it.

---

## Table of Contents

- [Paper Overview](#paper-overview)
- [Implementation Details](#implementation-details)
- [Network Architecture](#network-architecture)
- [Training Pipeline](#training-pipeline)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Command Line Arguments](#command-line-arguments)
- [Experimental Results](#experimental-results)
- [References](#references)

## Paper Overview

**DEC** (Deep Embedded Clustering) is an unsupervised clustering method that combines deep learning with traditional clustering algorithms, learning low-dimensional embeddings of data while optimizing clustering objectives.

**Core Idea**:
1. Use a Stacked AutoEncoder (SAE) for pretraining to obtain initial data embeddings
2. Initialize cluster centers using K-Means
3. Iteratively optimize the KL divergence loss, updating both network parameters and cluster centers

## Implementation Details

### 1. Pretraining Phase

Learn compressed representations using a stacked autoencoder, supporting two pretraining approaches:

- **Layer-wise Pretraining (Recommended)**: Train multiple shallow autoencoders sequentially, where each layer takes features extracted by the previous encoder as input. After training, stack them into a deep network. This effectively mitigates gradient vanishing in deep network training.
- **End-to-end Pretraining**: Train the entire stacked autoencoder directly. Simple but may not converge as well as layer-wise pretraining.

### 2. Fine-tuning Phase

Remove Dropout regularization and continue training the autoencoder to optimize feature representation quality.

### 3. Clustering Phase

Perform unsupervised clustering using Deep Embedded Clustering:

- **Soft Assignment**: Compute similarity between samples and cluster centers based on t-distribution
- **Target Distribution**: Calculate target distribution by amplifying high-confidence assignments and suppressing low-confidence ones
- **Loss Function**: Minimize KL divergence between soft assignments and target distribution

## Network Architecture

```
Input (28×28=784)
    ↓
Encoder: 784 → 500 → 500 → 2000 → 10 (bottleneck, linear)
    ↓
Decoder: 10 → 2000 → 500 → 500 → 784 (output with Sigmoid)
    ↓
Output (784)
```

**Note**: The bottleneck layer remains linear, while the decoder output layer uses Sigmoid to match pixel range `[0,1]`.

## Training Pipeline

The complete training pipeline consists of three phases:

### Phase 1: SAE Pretraining
- Train autoencoder with Dropout regularization
- Supports layer-wise (`--layerwise`) and end-to-end pretraining modes
- Uses `CSVLogger` to record training loss in `logs/sae_pretrain/`

### Phase 2: SAE Fine-tuning
- Remove Dropout, continue training to optimize feature representations
- Uses `EarlyStopping` mechanism to prevent overfitting
- Saves model weights to `./checkpoints/sae_finetuned.ckpt` after training

### Phase 3: DEC Clustering Training
- Initialize cluster centers using K-Means
- Iteratively optimize KL divergence loss
- Supports gradient clipping (`gradient_clip_val=0.5`) for improved training stability
- Saves model weights to `./checkpoints/dec_trained.ckpt` after training

## Project Structure

```
dec-pytorch/
├── main.py              # Main training script
├── model.py             # SAE and DEC model definitions
├── utils.py             # Soft cluster assignment layer and helper functions
├── README.md            # Project documentation
├── checkpoints/         # Model checkpoint directory
│   ├── sae_finetuned.ckpt
│   └── dec_trained.ckpt
├── logs/                # Training logs directory
│   ├── sae_pretrain/
│   ├── sae_finetune/
│   └── dec/
└── MNIST/               # Auto-downloaded MNIST dataset
```

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Training entry point with argument parsing, SAE pretraining/fine-tuning, and DEC training |
| `model.py` | Implementation of SAE, AutoEncoder, DEC, and `cluster_acc` evaluation function |
| `utils.py` | `SoftClusterAssignment` class based on t-distribution |

## Environment Setup

### Dependencies

```
torch >= 2.0
torchvision >= 0.15
pytorch-lightning >= 2.0
scipy >= 1.10
scikit-learn >= 1.3
numpy >= 1.24
matplotlib >= 3.5
tensorboard >= 2.0
```

### Installation

```bash
pip install torch torchvision pytorch-lightning scipy scikit-learn numpy matplotlib tensorboard
```

### CUDA Support

For GPU acceleration, install CUDA-enabled PyTorch:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Full Training (Layer-wise Pretraining + DEC)

Run the complete DEC training pipeline (recommended):

```bash
python main.py
```

### Full Training (End-to-end Pretraining + DEC)

Use end-to-end pretraining instead of layer-wise:

```bash
python main.py --no-layerwise
```

### Train Autoencoder Only

If you only need to train the stacked autoencoder (without clustering):

```bash
python main.py --mode sae
```

### Load Pretrained Models

Skip training and load pretrained weights directly:

```bash
python main.py --sae_pretrained ./checkpoints/sae_finetuned.ckpt
```

```bash
python main.py --sae_pretrained ./checkpoints/sae_finetuned.ckpt --dec_pretrained ./checkpoints/dec_trained.ckpt
```

### Quick Test

Run a quick test with fewer epochs:

```bash
python main.py --epoch_pretrain 10 --epoch_finetune 10 --epoch_dec 20
```

## Command Line Arguments

### Basic Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | 711 | Random seed for reproducibility |
| `--mode` | dec | Training mode: `sae` for autoencoder only, `dec` for full pipeline |
| `--patience` | 50 | Patience for early stopping |

### SAE Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--layerwise` | True | Use layer-wise pretraining |
| `--epoch_pretrain` | 300 | Maximum epochs for pretraining |
| `--epoch_finetune` | 500 | Maximum epochs for fine-tuning |
| `--activation` | ReLU() | Activation function |
| `--dropout` | 0.2 | Dropout rate |
| `--batch_size` | 256 | Batch size |
| `--lr` | 1.0 | Learning rate |
| `--lr_decay` | 0.1 | Learning rate decay factor |
| `--lr_decay_step` | 20000 | Learning rate decay step (iterations) |
| `--weight_decay` | 0.0 | Weight decay coefficient |
| `--opt` | SGD | Optimizer: `SGD` or `Adam` |
| `--drop_last` | False | Drop the last incomplete batch |

### DEC Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epoch_dec` | 200 | Maximum epochs for DEC training |
| `--lr_dec` | 0.01 | Learning rate for DEC |
| `--update_interval` | 274 | Target distribution update interval |
| `--tol` | 0.001 | Tolerance for stopping criterion |
| `--maxiter` | 20000 | Maximum iterations for DEC training |

### Model Save/Load

| Argument | Default | Description |
|----------|---------|-------------|
| `--sae_pretrained` | None | Path to pretrained SAE model; skips SAE training if set |
| `--finetuned` | True | Whether to fine-tune after loading SAE model |
| `--dec_pretrained` | None | Path to pretrained DEC model; skips DEC training if set |
| `--sae_save_path` | `./checkpoints/sae_finetuned.ckpt` | Save path for fine-tuned SAE |
| `--dec_save_path` | `./checkpoints/dec_trained.ckpt` | Save path for trained DEC |

## Experimental Results

Comparison on MNIST dataset:

| Method | ACC (%) | NMI | ARI |
|--------|---------|-----|-----|
| Standard K-Means | ~53% | ~0.50 | ~0.40 |
| DEC (Original Paper) | 84.0% | — | — |
| **This Repository (10-dim)** | **87.38%** | **0.843** | **0.808** |

> This implementation achieves state-of-the-art performance at this bottleneck dimension through multiple critical fixes.

## References

- **Paper**: [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335) (ICML 2016)
- **Reference Implementation**: [youngerous/dec-pytorch](https://github.com/youngerous/dec-pytorch)
- **Original Implementation**: [piiswrong/dec](https://github.com/piiswrong/dec)

---

**License**  
This repository retains the license of the original reference code.

**Acknowledgements**  
Thanks to [youngerous/dec-pytorch](https://github.com/youngerous/dec-pytorch) for providing the baseline code, upon which extensive debugging and optimizations were performed to achieve improved performance.