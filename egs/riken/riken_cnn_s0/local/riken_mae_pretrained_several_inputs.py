"""
Train Masked Autoencoder (MAE) model for self-supervised learning on multiple spectrogram datasets.
Supports joint training from multiple input sources with visualization of all datasets.
Compatible with riken_mae_vit_train_v1.py for downstream tasks.

The script includes:
- MAE pretraining with configurable mask ratio on multiple datasets
- Support for different pretraining initialization options
- TensorBoard monitoring of losses and reconstructions from all input sources
- Regular checkpointing and best model saving
- Compatible with downstream ViT training
- Fixed NaN loss issues with mixed precision training
- Enhanced visualization to ensure all datasets appear in TensorBoard

Author: bin-wu
Date: December 27, 2024 (Modified for multiple inputs with enhanced visualization)
"""

import os
import sys
import glob
import shutil
import datetime
import logging
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import ViTMAEForPreTraining, ViTMAEConfig
import GPUtil

def set_seed(seed):
    """
    Set random seeds for reproducibility across NumPy and PyTorch.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(gpu):
    """
    Determine the device (GPU or CPU) based on the specified GPU argument.
    """
    if gpu == 'auto':
        # Get the available GPU with the least memory usage
        available_gpus = GPUtil.getAvailable(order='memory')
        if available_gpus:
            device = torch.device(f"cuda:{available_gpus[0]}")
        else:
            device = torch.device("cpu")
    else:
        # Use the specified GPU device if available, otherwise use CPU
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device

def init_logger(file_name="", stream="stdout"):
    """
    Initialize a logger to terminal and file at the same time.

    Args:
        file_name (str): Path to log file. If empty, only log to terminal
        stream (str): Stream type, either "stdout" or "stderr"

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s",
                                "%d/%m/%Y %H:%M:%S")

    logger.handlers = [] # Clear existing stream and file handlers
    if stream == "stdout":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_name:
        file_handler = logging.FileHandler(file_name, 'w') # overwrite the log file if exists
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def continue_train(message=""):
    """
    Prompts the user to decide whether to continue training and for how many additional epochs.

    Args:
        message (str): Optional message to display before the prompt

    Returns:
        tuple: (bool: continue training, int: additional epochs)
    """
    continue_or_not = ""
    while continue_or_not not in {'yes', 'y', 'no', 'n'}:
        continue_or_not = input(message + "Continue to train [y/n]?").lower().strip()

    add_epochs = "0" if continue_or_not in {'no', 'n'} else ""

    while not add_epochs.isdigit():
        add_epochs = input("How many additional epochs [1 to N]:").lower().strip()

    return continue_or_not in {'yes', 'y'}, int(add_epochs)

def overwrite_result_directory(result_dir, overwrite=False):
    """
    Prompts the user to decide whether to overwrite the result directory or not.

    Args:
        result_dir (str): Path to the result directory
        overwrite (bool): If True, overwrite without prompting
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        overwrite_or_not = 'yes' if overwrite else None
        while overwrite_or_not not in {'yes', 'no', 'n', 'y'}:
            overwrite_or_not = input(f"Overwriting the result directory ('{result_dir}') [y/n]?").lower().strip()

        if overwrite_or_not in {'yes', 'y'}:
            for x in glob.glob(os.path.join(result_dir, "*")):
                if os.path.isdir(x):
                    shutil.rmtree(x)
                if os.path.isfile(x):
                    os.remove(x)
            print(f"!!!Overwriting the result directory: '{result_dir}'")
        else:
            sys.exit(0)

class SpectrogramDataset(Dataset):
    """
    Dataset for loading and preprocessing spectrogram data for MAE training.

    Args:
        input_path (str): Path to input .npy file containing spectrograms
        dataset_name (str): Name identifier for the dataset (for logging/visualization)
        transform (callable, optional): Additional transforms to apply

    Attributes:
        inputs (np.ndarray): Loaded spectrogram data
        dataset_name (str): Dataset identifier
        transform (callable): Transform pipeline
        normalize (transforms.Normalize): ImageNet normalization
    """
    def __init__(self, input_path, dataset_name=None, transform=None):
        self.inputs = np.load(input_path)
        self.dataset_name = dataset_name or os.path.basename(input_path)
        self.transform = transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def process_input(self, x):
        """Process single-channel spectrogram input"""
        # Convert to float32 if needed
        x = x.astype(np.float32)

        # Normalize to [0, 1] with small epsilon to avoid division by zero
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = x - x_min  # All values are the same, just center at 0

        # Add channel dimension and convert to tensor
        x = torch.from_numpy(x).unsqueeze(0)

        # Convert to 3 channels
        x = x.repeat(3, 1, 1)

        # Apply resize transform if any
        if self.transform:
            x = self.transform(x)

        # Apply ImageNet normalization
        x = self.normalize(x)

        return x

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        return self.process_input(x)

class MultiDatasetWrapper(Dataset):
    """
    Wrapper for ConcatDataset that tracks which dataset each sample comes from.
    This allows us to visualize samples from each dataset separately.

    Args:
        datasets (list): List of SpectrogramDataset objects
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)

        # Create mapping from global index to (dataset_idx, local_idx)
        self.dataset_offsets = []
        cumsum = 0
        for dataset in datasets:
            self.dataset_offsets.append(cumsum)
            cumsum += len(dataset)
        self.dataset_offsets.append(cumsum)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        # Get the actual data
        data = self.concat_dataset[idx]

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i in range(len(self.dataset_offsets) - 1):
            if self.dataset_offsets[i] <= idx < self.dataset_offsets[i + 1]:
                dataset_idx = i
                break

        return data, dataset_idx

def create_balanced_sampler(dataset_wrapper, samples_per_dataset=32):
    """
    Create a sampler that ensures each dataset is represented in early batches

    Args:
        dataset_wrapper: MultiDatasetWrapper instance
        samples_per_dataset: Number of samples from each dataset to put at the beginning

    Returns:
        list: List of indices with balanced representation at the start
    """
    indices = []
    dataset_offsets = dataset_wrapper.dataset_offsets

    # Add samples from each dataset to the beginning
    actual_samples_per_dataset = min(samples_per_dataset,
                                   min(len(d) for d in dataset_wrapper.datasets))

    for i, dataset in enumerate(dataset_wrapper.datasets):
        start_idx = dataset_offsets[i]
        dataset_indices = list(range(start_idx, start_idx + actual_samples_per_dataset))
        indices.extend(dataset_indices)

    # Add the rest of the indices
    all_indices = set(range(len(dataset_wrapper)))
    used_indices = set(indices)
    remaining_indices = list(all_indices - used_indices)

    indices.extend(remaining_indices)
    return indices

def setup_optimizer_and_scheduler(model, base_lr, batch_size, weight_decay, warmup_epochs, total_epochs, constant_base_lr=True):
    """
    Set up optimizer and learning rate scheduler with warmup and cosine decay.

    Args:
        model (nn.Module): The PyTorch model to optimize
        base_lr (float): Base learning rate
        batch_size (int): Training batch size
        weight_decay (float): Weight decay factor for L2 regularization
        warmup_epochs (int): Number of epochs for linear warmup
        total_epochs (int): Total number of epochs for training
        constant_base_lr (bool): If True, use fixed base_lr; if False, scale with batch size

    Returns:
        tuple: (optimizer, scheduler)
            - optimizer (torch.optim.AdamW): The initialized optimizer
            - scheduler (torch.optim.lr_scheduler.SequentialLR): Combined warmup and
              cosine annealing scheduler

    Note:
        If constant_base_lr is False, learning rate is scaled:
        actual_lr = base_lr * (batch_size / 256)
    """
    actual_lr = base_lr if constant_base_lr else base_lr * (batch_size / 256)

    optimizer = AdamW(model.parameters(), lr=actual_lr, weight_decay=weight_decay)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    return optimizer, scheduler

def create_mae_model(args, logger):
    """
    Create MAE model with specified configuration.

    Args:
        args: Command line arguments containing:
            - image_size (int): Input image size
            - patch_size (int): Size of patches
            - seq (bool): Whether to use 1D sequential patching
            - dim (int): Hidden dimension size
            - depth (int): Number of transformer layers
            - heads (int): Number of attention heads
            - mlp_dim (int): Dimension of MLP layer
            - dropout (float): Dropout rate
            - mask_ratio (float): Ratio of patches to mask
            - pretrained_path (str): Path to pretrained model, 'base' for facebook/vit-mae-base, or None

    Returns:
        ViTMAEForPreTraining: Configured MAE model
    """
    if args.pretrained_path == 'base':
        config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
        config.mask_ratio = args.mask_ratio  # Override mask ratio
        model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base', config=config)
    elif args.pretrained_path:
        # Load from checkpoint saved by riken_mae_pretrained.py
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        # Create a new MAE model first
        config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
        config.mask_ratio = args.mask_ratio  # Override mask ratio
        model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base', config=config)
        # Load the saved state dict
        model.load_state_dict(checkpoint['model'])
        # Update mask ratio in model's config
        model.config.mask_ratio = args.mask_ratio
    else:
        config = ViTMAEConfig(
            image_size=args.image_size,
            patch_size=args.patch_size if not args.seq else (args.image_size, args.patch_size),
            num_channels=3,
            hidden_size=args.dim,
            num_hidden_layers=args.depth,
            num_attention_heads=args.heads,
            intermediate_size=args.mlp_dim,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            mask_ratio=args.mask_ratio
        )
        model = ViTMAEForPreTraining(config)

    logger.info(f"Model created with mask_ratio: {model.config.mask_ratio}")
    return model

def check_for_nan_inf(tensor, name="tensor"):
    """Check for NaN or Inf values in tensors and log if found"""
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")
        return True
    return False

def collect_samples_by_dataset_enhanced(data_batch, dataset_indices, num_datasets, num_save_images, existing_samples=None):
    """
    Enhanced function to organize samples by their source dataset for visualization.
    Continues collecting until we have samples from all datasets.

    Args:
        data_batch: Batch of data tensors
        dataset_indices: Batch of dataset indices
        num_datasets: Total number of datasets
        num_save_images: Number of images to save per dataset
        existing_samples: Previously collected samples (for continuation)

    Returns:
        dict: Mapping from dataset_idx to tensor of samples from that dataset
        bool: Whether we have samples from all datasets
    """
    if existing_samples is None:
        samples_by_dataset = {}
    else:
        samples_by_dataset = existing_samples.copy()

    for dataset_idx in range(num_datasets):
        # Skip if we already have enough samples for this dataset
        if dataset_idx in samples_by_dataset and len(samples_by_dataset[dataset_idx]) >= num_save_images:
            continue

        # Find samples from this dataset
        mask = dataset_indices == dataset_idx
        if mask.sum() > 0:
            dataset_samples = data_batch[mask]

            if dataset_idx not in samples_by_dataset:
                # Take up to num_save_images samples
                n_samples = min(len(dataset_samples), num_save_images)
                samples_by_dataset[dataset_idx] = dataset_samples[:n_samples]
            else:
                # Add more samples if we don't have enough yet
                existing = samples_by_dataset[dataset_idx]
                needed = num_save_images - len(existing)
                if needed > 0:
                    additional = dataset_samples[:needed]
                    samples_by_dataset[dataset_idx] = torch.cat([existing, additional], dim=0)

    # Check if we have samples from all datasets
    all_datasets_covered = len(samples_by_dataset) == num_datasets

    return samples_by_dataset, all_datasets_covered

def train_mae(args, logger):
    """
    Main training function for MAE model with multiple input datasets.

    Args:
        args: Command line arguments containing:
            - gpu (str): GPU selection
            - batch_size (int): Batch size for training
            - num_workers (int): Number of data loading workers
            - train_input (list): List of paths to training data
            - dev_input (list): List of paths to validation data
            - output_dir (str): Directory for saving outputs
            - base_lr (float): Base learning rate
            - weight_decay (float): Weight decay factor
            - warmup_epochs (int): Number of warmup epochs
            - total_epochs (int): Total epochs for training
            - num_epochs (int): Current number of epochs to train
            - save_epoch_interval (int): Interval for saving checkpoints
            - save_image_interval (int): Interval for saving images
            - num_save_images (int): Number of images to save
            - exit (bool): Whether to exit after training
            - disable_amp (bool): Whether to disable automatic mixed precision
        logger: Logger instance for recording training progress
    """
    device = set_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Create model
    model = create_mae_model(args, logger).to(device)
    logger.info("Model created")

    transform = transforms.Resize((args.image_size, args.image_size))

    # Create individual datasets for train and dev
    train_datasets = []
    for i, train_path in enumerate(args.train_input):
        dataset_name = f"train_dataset_{i}"
        dataset = SpectrogramDataset(train_path, dataset_name=dataset_name, transform=transform)
        train_datasets.append(dataset)
        logger.info(f"Train dataset {i} ({dataset_name}) size: {len(dataset)}")

    dev_datasets = []
    for i, dev_path in enumerate(args.dev_input):
        dataset_name = f"dev_dataset_{i}"
        dataset = SpectrogramDataset(dev_path, dataset_name=dataset_name, transform=transform)
        dev_datasets.append(dataset)
        logger.info(f"Dev dataset {i} ({dataset_name}) size: {len(dataset)}")

    # Create multi-dataset wrappers
    train_multi_dataset = MultiDatasetWrapper(train_datasets)
    val_multi_dataset = MultiDatasetWrapper(dev_datasets)

    # Create balanced sampler for validation to ensure all datasets appear in early batches
    val_sampler_indices = create_balanced_sampler(val_multi_dataset)

    train_loader = DataLoader(
        train_multi_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_multi_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_sampler_indices),
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"Combined train dataset size: {len(train_multi_dataset)}")
    logger.info(f"Combined validation dataset size: {len(val_multi_dataset)}")

    # Debug: Check validation dataset distribution in first few batches
    logger.info("Checking validation dataset distribution in first few batches:")
    with torch.no_grad():
        for batch_idx, (batch, dataset_indices) in enumerate(val_loader):
            unique_datasets = torch.unique(dataset_indices)
            logger.info(f"Validation batch {batch_idx}: Contains samples from datasets {unique_datasets.tolist()}")
            if batch_idx >= 2:  # Check first 3 batches
                break

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model=model,
        base_lr=args.base_lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs
    )

    # Setup tensorboard and gradient scaler
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Initialize scaler only if AMP is enabled
    if not args.disable_amp:
        scaler = GradScaler()
        logger.info("Using Automatic Mixed Precision")
    else:
        scaler = None
        logger.info("AMP disabled - using full precision")

    # Training loop
    best_val_loss = float('inf')
    epoch = 0

    def unpatchify(x, patch_size=16):
        """Convert patches back to images"""
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        h = w = int(num_patches ** 0.5)
        channels = 3

        # Reshape to [B, h, w, p, p, C]
        x = x.reshape(batch_size, h, w, patch_size, patch_size, channels)

        # Permute and reshape to image format [B, C, H, W]
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, channels, h*patch_size, w*patch_size)
        return x

    def visualize_spectrogram(x):
        """
        Convert tensor to spectrogram visualization
        Args:
            x: tensor of shape (batch_size, 3, H, W)
        Returns:
            tensor of shape (batch_size, 3, H, W)
        """
        # Normalize to [0,1] with clipping to handle potential NaN/Inf
        x = torch.clamp(x, min=-1e6, max=1e6)  # Clamp extreme values
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = torch.zeros_like(x)
        return x

    while epoch < args.num_epochs:
        model.train()
        train_loss = 0
        train_samples_by_dataset = {}
        train_recon_by_dataset = {}
        nan_count = 0

        # Training phase
        train_samples_collected = False
        for batch_idx, (batch, dataset_indices) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')):
            images = batch.to(device)

            # Check for NaN/Inf in inputs
            if check_for_nan_inf(images, "input_images"):
                logger.warning(f"Skipping batch {batch_idx} due to NaN/Inf in inputs")
                continue

            if args.disable_amp:
                # Full precision training
                outputs = model(images)
                loss = outputs.loss
            else:
                # Mixed precision training
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = outputs.loss

            # Check for NaN/Inf in loss
            if check_for_nan_inf(loss, "loss"):
                nan_count += 1
                logger.warning(f"NaN/Inf detected in loss at batch {batch_idx}, epoch {epoch}")
                if nan_count > 10:  # Skip if too many NaN
                    logger.error("Too many NaN losses, skipping epoch")
                    break
                continue

            train_loss += loss.item()

            optimizer.zero_grad()

            if args.disable_amp:
                # Full precision backward
                loss.backward()
                optimizer.step()
            else:
                # Mixed precision backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Collect samples for visualization until we have samples from all datasets
            if not train_samples_collected:
                # Organize samples by dataset
                new_samples, all_covered = collect_samples_by_dataset_enhanced(
                    images.detach(), dataset_indices, len(train_datasets),
                    args.num_save_images, train_samples_by_dataset
                )
                train_samples_by_dataset = new_samples

                # Get reconstructions for each dataset
                with torch.no_grad():
                    recon = unpatchify(outputs.logits.detach())
                    new_recon, _ = collect_samples_by_dataset_enhanced(
                        recon, dataset_indices, len(train_datasets),
                        args.num_save_images, train_recon_by_dataset
                    )
                    train_recon_by_dataset = new_recon

                # Stop collecting if we have samples from all datasets
                train_samples_collected = all_covered
                if train_samples_collected:
                    logger.info(f"Training: Successfully collected samples from all {len(train_datasets)} datasets")

        if nan_count > 0:
            logger.warning(f"Encountered {nan_count} NaN losses in epoch {epoch}")

        # Calculate average training loss
        num_valid_batches = len(train_loader) - nan_count
        if num_valid_batches > 0:
            train_loss /= num_valid_batches
        else:
            train_loss = float('nan')

        # Save training reconstructions by dataset
        if epoch % args.save_image_interval == 0:
            for dataset_idx in range(len(train_datasets)):
                dataset_name = train_datasets[dataset_idx].dataset_name
                if dataset_idx in train_samples_by_dataset:
                    writer.add_images(
                        f'Train/{dataset_name}/Original',
                        visualize_spectrogram(train_samples_by_dataset[dataset_idx]),
                        epoch, dataformats='NCHW'
                    )
                if dataset_idx in train_recon_by_dataset:
                    writer.add_images(
                        f'Train/{dataset_name}/Reconstructed',
                        visualize_spectrogram(train_recon_by_dataset[dataset_idx]),
                        epoch, dataformats='NCHW'
                    )

        # Validation
        model.eval()
        val_loss = 0
        val_samples_by_dataset = {}
        val_recon_by_dataset = {}
        val_nan_count = 0

        with torch.no_grad():
            val_samples_collected = False
            for batch_idx, (batch, dataset_indices) in enumerate(val_loader):
                images = batch.to(device)

                # Check for NaN/Inf in inputs
                if check_for_nan_inf(images, "val_input_images"):
                    continue

                if args.disable_amp:
                    outputs = model(images)
                else:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(images)

                # Check for NaN/Inf in loss
                if check_for_nan_inf(outputs.loss, "val_loss"):
                    val_nan_count += 1
                    continue

                val_loss += outputs.loss.item()

                # Collect samples for visualization until we have samples from all datasets
                if not val_samples_collected:
                    # Organize samples by dataset
                    new_samples, samples_all_covered = collect_samples_by_dataset_enhanced(
                        images.detach(), dataset_indices, len(dev_datasets),
                        args.num_save_images, val_samples_by_dataset
                    )
                    val_samples_by_dataset = new_samples

                    # Get reconstructions for each dataset
                    recon = unpatchify(outputs.logits.detach())
                    new_recon, recon_all_covered = collect_samples_by_dataset_enhanced(
                        recon, dataset_indices, len(dev_datasets),
                        args.num_save_images, val_recon_by_dataset
                    )
                    val_recon_by_dataset = new_recon

                    # Stop collecting if we have samples from all datasets
                    val_samples_collected = samples_all_covered and recon_all_covered

                    # Debug logging
                    if batch_idx < 5:  # Only log for first few batches
                        logger.info(f"Validation batch {batch_idx}: Collected samples from datasets {list(val_samples_by_dataset.keys())}")
                    if val_samples_collected:
                        logger.info(f"Validation: Successfully collected samples from all {len(dev_datasets)} datasets")

        # Calculate average validation loss
        num_valid_val_batches = len(val_loader) - val_nan_count
        if num_valid_val_batches > 0:
            val_loss /= num_valid_val_batches
        else:
            val_loss = float('nan')

        # Save validation reconstructions by dataset
        if epoch % args.save_image_interval == 0:
            for dataset_idx in range(len(dev_datasets)):
                dataset_name = dev_datasets[dataset_idx].dataset_name
                if dataset_idx in val_samples_by_dataset:
                    writer.add_images(
                        f'Val/{dataset_name}/Original',
                        visualize_spectrogram(val_samples_by_dataset[dataset_idx]),
                        epoch, dataformats='NCHW'
                    )
                    logger.info(f"Saved validation original images for {dataset_name}")
                if dataset_idx in val_recon_by_dataset:
                    writer.add_images(
                        f'Val/{dataset_name}/Reconstructed',
                        visualize_spectrogram(val_recon_by_dataset[dataset_idx]),
                        epoch, dataformats='NCHW'
                    )
                    logger.info(f"Saved validation reconstructed images for {dataset_name}")

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                   f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save model checkpoint
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(args.output_dir, f'model_epoch_{epoch}.pt')
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),  # Save state_dict instead of whole model
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': model.config,  # Save config for reference
            }
            if not args.disable_amp:
                checkpoint['scaler'] = scaler.state_dict()
            torch.save(checkpoint, save_path)
            logger.info(f"Saved model checkpoint to {save_path}")

        # Save best model
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'model_best.pt')
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),  # Save state_dict instead of whole model
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'best_val_loss': best_val_loss
            }
            if not args.disable_amp:
                checkpoint['scaler'] = scaler.state_dict()
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation loss {best_val_loss:.4f}")

        scheduler.step()

        # Continue training prompt
        if epoch == args.num_epochs - 1 and not args.exit:
            command = "python " + ' '.join([x for x in sys.argv])
            message = f"Command: '{command}'\nOutput directory: '{args.output_dir}'\n"
            continue_or_not, add_epochs = continue_train(message)
            if continue_or_not and add_epochs:
                args.num_epochs += add_epochs
                logger.info(f"Adding {add_epochs} more epochs")

        epoch += 1

    writer.close()
    logger.info("Training completed")

def main():
    parser = argparse.ArgumentParser(description='Train MAE model on multiple spectrogram datasets')

    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--train_input', type=str, nargs='+', required=True,
                           help='Paths to the input files for training sets (multiple allowed)')
    data_group.add_argument('--dev_input', type=str, nargs='+', required=True,
                           help='Paths to the input files for development sets (multiple allowed)')
    data_group.add_argument('--batch_size', type=int, default=256,
                           help='Batch size for training and validation')
    data_group.add_argument('--num_workers', type=int, default=4,
                           help='Number of workers for data loading')

    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--image_size', type=int, default=224,
                            help='Image size (height, width)')
    model_group.add_argument('--patch_size', type=int, default=16,
                            help='Patch size (height, width)')
    model_group.add_argument('--seq', action='store_true',
                            help='1D sequential MAE with patch size (image_height_int, patch_size_int)')
    model_group.add_argument('--dim', type=int, default=768,
                            help='Hidden dimension size')
    model_group.add_argument('--depth', type=int, default=12,
                            help='Number of transformer layers')
    model_group.add_argument('--heads', type=int, default=12,
                            help='Number of attention heads')
    model_group.add_argument('--mlp_dim', type=int, default=3072,
                            help='Dimension of the MLP layer')
    model_group.add_argument('--channels', type=int, default=1,
                            help='Number of input channels')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')
    model_group.add_argument('--mask_ratio', type=float, default=0.75,
                            help='Ratio of patches to mask during training')
    model_group.add_argument('--pretrained_path', type=str, default=None,
                            help="Path to pretrained model, 'base' for facebook/vit-mae-base, or None")

    # Output/Visualization arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_dir', type=str, default='exp/mae_pretraining/test_multiple',
                             help='Base directory for outputs')
    output_group.add_argument('--use_timestamp', action='store_true',
                             help='Add timestamp prefix to output directory')
    output_group.add_argument('--overwrite', action='store_true',
                             help='Overwrite existing output directory without prompting')
    output_group.add_argument('--num_save_images', type=int, default=8,
                             help='Number of images to save in tensorboard per dataset')

    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--num_epochs', type=int, default=401,
                            help='Number of training epochs')
    train_group.add_argument('--save_epoch_interval', type=int, default=10,
                            help='Save the model every N epochs')
    train_group.add_argument('--save_image_interval', type=int, default=5,
                            help='Save reconstruction images every N epochs')
    train_group.add_argument('--disable_amp', action='store_true',
                            help='Disable automatic mixed precision training')

    # Optimizer arguments
    optimizer_group = parser.add_argument_group('Optimizer Configuration')
    optimizer_group.add_argument('--weight_decay', type=float, default=0.05,
                                help='Weight decay for AdamW optimizer')
    optimizer_group.add_argument('--warmup_epochs', type=int, default=40,
                                help='Number of warmup epochs')
    optimizer_group.add_argument('--total_epochs', type=int, default=400,
                                help='Total number of epochs for cosine annealing')
    optimizer_group.add_argument('--base_lr', type=float, default=1.5e-4,
                                help='Base learning rate')

    # Other arguments
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=str, default='auto',
                        help="GPU selection: number for specific GPU or 'auto' for least used")
    parser.add_argument('--exit', action='store_true',
                        help='Exit after training instead of prompting to continue')

    args = parser.parse_args()

    # Validate input arguments
    if len(args.train_input) != len(args.dev_input):
        raise ValueError("Number of train_input and dev_input files must match")

    # Set random seed
    set_seed(args.seed)

    # Create output directory with timestamp if requested
    if args.use_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(args.output_dir, timestamp)

    # Handle directory creation/overwriting
    overwrite_result_directory(args.output_dir, args.overwrite)
    os.makedirs(os.path.join(args.output_dir, 'tensorboard'), exist_ok=True)

    # Initialize logger
    logger = init_logger(os.path.join(args.output_dir, "report.log"))
    logger.info(f"Arguments: {args}")
    logger.info(f"Training with {len(args.train_input)} datasets:")
    for i, (train_path, dev_path) in enumerate(zip(args.train_input, args.dev_input)):
        logger.info(f"  Dataset {i}: Train={train_path}, Dev={dev_path}")

    # Start training
    train_mae(args, logger)

if __name__ == '__main__':
    main()
