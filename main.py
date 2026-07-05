import argparse
import os
import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from model import DEC, SAE

class NanDetector(Callback):
    def on_batch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}")
                trainer.should_stop = True
                break

def main(hparams):
    """Main function: execute the full pipeline of SAE pretraining, fine-tuning, and DEC clustering.

    Supported modes:
    1. Full training pipeline: SAE pretraining -> SAE fine-tuning -> DEC training
    2. Load pre-trained SAE: skip SAE training, load weights from file
    3. Load pre-trained DEC: skip all training, load weights from file
    """
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(hparams.seed)
    
    os.makedirs("./checkpoints", exist_ok=True)
    
    # ========== SAE training or loading ==========
    if hparams.sae_pretrained is not None:
        # Load SAE model from pretrained weights
        pprint.pprint("#########################################")
        pprint.pprint("#     Loading SAE from pretrained weights     #")
        pprint.pprint("#########################################")
        sae = SAE.load_from_checkpoint(hparams.sae_pretrained, strict=False)
        if hparams.finetuned == True:
            # Fine-tune stacked autoencoder (no Dropout)
            pprint.pprint("########################################")
            pprint.pprint("#     Fine-tuning stacked autoencoder     #")
            pprint.pprint("########################################")
            sae.set_dropout(0.0)
            trainer_sae_ft = pl.Trainer(
                max_epochs=hparams.epoch_finetune,
                callbacks=[NanDetector(), EarlyStopping(monitor="train_loss", mode="min", patience=hparams.patience)],
                deterministic=True,
                gradient_clip_val=0.5,
                logger=CSVLogger("logs", name="sae_finetune"),
                enable_checkpointing=False,
            )
            trainer_sae_ft.fit(sae)
            
            # Save fine-tuned SAE model
            sae.save(hparams.sae_save_path)
    else:
        # Pretrain stacked autoencoder
        pprint.pprint("#########################################")
        pprint.pprint("#     Pretraining stacked autoencoder     #")
        pprint.pprint("#########################################")
        
        if hparams.layerwise:
            # Layer-wise pretraining
            sae = SAE.build_from_layerwise_pretraining(
                dimensions=[28 * 28, 500, 500, 2000, 10],
                activation=hparams.activation,
                dropout=hparams.dropout,
                batch_size=hparams.batch_size,
                lr=hparams.lr,
                drop_last=hparams.drop_last,
                opt=hparams.opt,
                epoch_pretrain=hparams.epoch_pretrain,
                patience=hparams.patience,
            )
        else:
            # End-to-end pretraining (original approach)
            sae = SAE([28 * 28, 500, 500, 2000, 10], 
                      activation=hparams.activation, 
                      dropout=hparams.dropout,
                      lr=hparams.lr,
                      drop_last=hparams.drop_last, 
                      opt=hparams.opt
                      )
            trainer_sae_pt = pl.Trainer(
                max_epochs=hparams.epoch_pretrain,
                callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=hparams.patience)],
                deterministic=True,
                logger=CSVLogger("logs", name="sae_pretrain"),
                enable_checkpointing=False,
            )
            trainer_sae_pt.fit(sae)

        # Fine-tune stacked autoencoder (no Dropout)
        pprint.pprint("########################################")
        pprint.pprint("#     Fine-tuning stacked autoencoder     #")
        pprint.pprint("########################################")
        sae.set_dropout(0.0)
        trainer_sae_ft = pl.Trainer(
            max_epochs=hparams.epoch_finetune,
            callbacks=[NanDetector(), EarlyStopping(monitor="train_loss", mode="min", patience=hparams.patience)],
            deterministic=True,
            gradient_clip_val=0.5,
            logger=CSVLogger("logs", name="sae_finetune"),
            enable_checkpointing=False,
        )
        trainer_sae_ft.fit(sae)
        
        # Save fine-tuned SAE model
        sae.save(hparams.sae_save_path)
    
    # Visualize SAE reconstruction
    sae.visualize_reconstruction()
    
    # If SAE-only mode, return here
    if hparams.mode == "sae":
        return
    
    # ========== DEC training or loading ==========
    if hparams.dec_pretrained is not None:
        # Load DEC model from pretrained weights (inference only)
        pprint.pprint("###########################################")
        pprint.pprint("#     Loading DEC from pretrained weights     #")
        pprint.pprint("###########################################")
        checkpoint = torch.load(hparams.dec_pretrained, map_location='cpu', weights_only=True)
        dec = DEC(sae.encoder, 
                  num_cluster=10, 
                  hidden_dim=10, 
                  drop_last=hparams.drop_last, 
                  lr_dec=hparams.lr_dec, 
                  tol=hparams.tol, 
                  update_interval=hparams.update_interval, 
                  maxiter=hparams.maxiter)
        dec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        dec.assignment.load_state_dict(checkpoint['assignment_state_dict'])
        print(f"DEC model loaded from: {hparams.dec_pretrained}")
        return
    else:
        # Train Deep Embedded Clustering (DEC)
        pprint.pprint("###########################################")
        pprint.pprint("#     Training Deep Embedded Clustering     #")
        pprint.pprint("###########################################")

        # Initialize DEC with SAE encoder
        dec = DEC(
                sae.encoder, 
                num_cluster=10, 
                hidden_dim=10,
                drop_last=hparams.drop_last, 
                batch_size=hparams.batch_size,
                lr_dec=hparams.lr_dec, 
                tol=hparams.tol, 
                update_interval=hparams.update_interval,
                maxiter=hparams.maxiter)
        trainer_dec = pl.Trainer(
            max_epochs=hparams.epoch_dec,
            deterministic=True, 
            gradient_clip_val=0.5,
            logger=CSVLogger("logs", name="dec"),
            enable_checkpointing=False,
        )
        trainer_dec.fit(dec)
        
        # Save trained DEC model
        dec.save(hparams.dec_save_path)

    # ========== DEC evaluation ==========
    metrics = dec.evaluate()
    print(metrics) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=711, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--mode", type=str, default="dec", help="Training mode: 'sae' for SAE only, 'dec' for full DEC pipeline"
    )
    
    # Early stopping settings
    parser.add_argument(
        "--patience", type=int, default=50, help="Patience for early stopping"
    )
    
    # SAE settings
    parser.add_argument(
        "--layerwise", action="store_true", default=True, help="Use layer-wise pretraining"
    )
    
    parser.add_argument(
        "--epoch_pretrain", type=int, default=300, help="Maximum epochs for pretraining"
    )
    parser.add_argument(
        "--epoch_finetune", type=int, default=500, help="Maximum epochs for fine-tuning"
    )
    parser.add_argument("--activation", type=nn.Module, default=nn.ReLU(), help="Activation function, e.g. 'leaky_relu(0.1)' or 'relu()'")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--lr_decay_step",
        type=float,
        default=20000,
        help="Learning rate decay step (iterations)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay coefficient"
    )

    # DEC settings
    parser.add_argument(
        "--lr_dec", type=float, default=0.01, help="Learning rate for DEC training"
    )
    parser.add_argument(
        "--epoch_dec", type=int, default=200, help="Maximum epochs for DEC training"
    )
    parser.add_argument(
        "--update_interval", type=int, default=274, help="Update interval for DEC training"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.001,
        help="Tolerance for stopping criterion",
    )
    parser.add_argument(
        "--maxiter", type=int, default=20000, help="Maximum iterations for DEC training"
    )
    
    parser.add_argument(
        "--drop_last",
        type=bool,
        default=False,
        help="Whether to drop the last incomplete batch",
    )

    # Model weight save/load settings
    parser.add_argument(
        "--sae_pretrained",
        type=str,
        default=None,
        help="Path to pretrained SAE model (e.g., ./checkpoints/sae_finetuned.ckpt). Skips SAE pretraining and fine-tuning if set",
    )
    parser.add_argument(
        "--finetuned",
        type=bool,
        default=True,
        help="Whether to fine-tune after loading SAE model",
    )
    parser.add_argument(
        "--dec_pretrained",
        type=str,
        default=None,
        help="Path to pretrained DEC model. Skips DEC training if set",
    )
    parser.add_argument(
        "--sae_save_path",
        type=str,
        default="./checkpoints/sae_finetuned.ckpt",
        help="Path to save fine-tuned SAE model",
    )
    parser.add_argument(
        "--dec_save_path",
        type=str,
        default="./checkpoints/dec_trained.ckpt",
        help="Path to save trained DEC model",
    )

    hparams = parser.parse_args()
    main(hparams)