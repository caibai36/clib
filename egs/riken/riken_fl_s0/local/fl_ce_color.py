"""
Train MAE model for calculating conditional entropy H(X|Y) through image reconstruction.
Uses ImageNet data and MAE architecture without masking.

Author: bin-wu
Date: January 14, 2025
"""

import os
import sys

import logging
import glob
import shutil
import re

import random
import argparse

import GPUtil
import tqdm
import tabulate
from omegaconf import OmegaConf

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import ViTMAEForPreTraining, ViTMAEConfig

def set_seed(seed):
    """
    Set the seed for reproducibility across NumPy and PyTorch.

    Args:
        seed (int): Random seed value
    """
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

    Args:
        gpu (str): GPU selection ('auto' or specific GPU id)

    Returns:
        torch.device: Selected device
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
    """Initialize a logger to terminal and file at the same time."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", "%d/%m/%Y %H:%M:%S")

    logger.handlers = []
    if stream == "stdout":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_name:
        file_handler = logging.FileHandler(file_name, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def continue_train(message=""):
    """
    Prompts the user to decide whether to continue training and for how many additional epochs.

    Args:
        message (str): Optional message to display before the prompt. Default is an empty string.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the user decides to continue training, False otherwise.
            - int: The number of additional epochs to train, as specified by the user.

    Example:
        >>> continue_training, additional_epochs = continue_train("Epoch 10 completed. ")
        Epoch 10 completed. Continue to train [y/n]? y
        How many additional epochs [1 to N]: 5
        >>> continue_training
        True
        >>> additional_epochs
        5
    """
    continue_or_not = ""
    # Prompt the user to decide whether to continue training
    # Valid responses are 'yes', 'y', 'no', or 'n' (case-insensitive)
    while continue_or_not not in {'yes', 'y', 'no', 'n'}:
        continue_or_not = input(message + "Continue to train [y/n]?").lower().strip()

    # If the user decides not to continue training, set the number of additional epochs to 0
    add_epochs = "0" if continue_or_not in {'no', 'n'} else ""

    # Prompt the user to enter the number of additional epochs to train
    while not add_epochs.isdigit():
        add_epochs = input("How many additional epochs [1 to N]:").lower().strip()

    # Return a tuple containing:
    # - True if the user decides to continue training (i.e., response is 'yes' or 'y'), False otherwise
    # - The number of additional epochs to train, as an integer
    return continue_or_not in {'yes', 'y'}, int(add_epochs)

def overwrite_result_directory(result_dir, overwrite=False):
    """
    Prompts the user to decide whether to overwrite the result directory or not when it exists.
    If the result directory does not exist, make the directory

    Args:
        result_dir (str): The path to the result directory.
        overwrite (bool, optional): If True, overwrite the result directory without prompting the user.
                                    Default is False.
    Example:
        >>> result_directory = 'output/experiment1'
        >>> overwrite_result_directory(result_directory)
        Overwriting the result directory ('output/experiment1') [y/n]? y
        !!!Overwriting the result directory: 'output/experiment1'
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        overwrite_or_not = 'yes' if overwrite else None
        # Prompt the user to decide whether to overwrite the result directory
        # Valid responses are 'yes', 'y', 'no', or 'n' (case-insensitive)
        while overwrite_or_not not in {'yes', 'no', 'n', 'y'}:
            overwrite_or_not = input(f"Overwriting the result directory ('{result_dir}') [y/n]?").lower().strip()

        if overwrite_or_not in {'yes', 'y'}:
            # If the user confirms to overwrite the result directory
            for x in glob.glob(os.path.join(result_dir, "*")):
                if os.path.isdir(x):
                    shutil.rmtree(x)  # Remove subdirectories recursively
                if os.path.isfile(x):
                    os.remove(x)  # Remove files
            overwrite_warning = f"!!!Overwriting the result directory: '{result_dir}'"
            print(overwrite_warning)
        else:
            # If the user decides not to overwrite the result directory
            sys.exit(0)  # Exit the script with a status code of 0

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Mean and variance come from imagenet statistics.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        # Get all image files and sort them
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]
        self.image_files.sort(key=lambda x: int(re.findall(r'(\d+)', x)[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # For validation set, you might want to extract the class ID from filename
        # or use a separate label file if available
        # Here we're just returning a dummy label (0) as placeholder
        label = 0

        return image, label

def create_mae_model(*, image_size=224, patch_size=16, dim=768, depth=12,
                    heads=12, mlp_dim=3072, dropout=0.0, channels=3,
                     mae_pretrained='base', mae_train_layers=-1, logger=logging):
    """
    Create and configure MAE model for image reconstruction without masking.

    This function:
    - Creates MAE model with masking disabled (mask_ratio=0)
    - Supports loading from pretrained checkpoints
    - Handles layer freezing for transfer learning
    - Maintains reconstruction-focused architecture

    Args:
        image_size (int, optional): Input image size (assumes square images). Defaults to 224.
        patch_size (int, optional): Size of image patches. Defaults to 16.
        dim (int, optional): Hidden dimension size. Defaults to 768.
        depth (int, optional): Number of transformer layers. Defaults to 12.
        heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_dim (int, optional): Dimension of MLP layer. Defaults to 3072.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        channels (int, optional): Number of input channels. Defaults to 3.
        mae_pretrained (str, optional): Path to pretrained model ('base' for facebook/vit-mae-base,
                                     or path to saved checkpoint). Defaults to 'base'.
        mae_train_layers (int, optional): Number of last layers to keep trainable.
                                       -1 means all layers trainable. Defaults to -1.
        logger (logging.Logger, optional): Logger instance for output. Defaults to None.

    Returns:
        ViTMAEForPreTraining: Configured MAE model for reconstruction

    Notes:
        - When using pretrained models, only the specified number of last layers will be trainable
        - mask_ratio is always set to 0 as we're doing pure reconstruction without masking
    """
    if mae_pretrained == 'base':
        # Initialize from pretrained facebook/vit-mae-base
        model = ViTMAEForPreTraining.from_pretrained(
            'facebook/vit-mae-base',
            config=ViTMAEConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                intermediate_size=mlp_dim,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                num_channels=channels,
                mask_ratio=0.0  # Disable masking for reconstruction
            )
        )
        if logger:
            logger.info("Loaded pretrained facebook/vit-mae-base model")

    elif mae_pretrained:
        # Load from custom checkpoint
        checkpoint = torch.load(mae_pretrained, map_location='cpu')
        model = ViTMAEForPreTraining.from_pretrained(
            'facebook/vit-mae-base',
            config=ViTMAEConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                intermediate_size=mlp_dim,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                num_channels=channels,
                mask_ratio=0.0
            )
        )
        model.load_state_dict(checkpoint['model'])
        if logger:
            logger.info(f"Loaded pretrained model from {mae_pretrained}")

    else:
        # Initialize fresh model
        model = ViTMAEForPreTraining(
            config=ViTMAEConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                intermediate_size=mlp_dim,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                num_channels=channels,
                mask_ratio=0.0
            )
        )
        if logger:
            logger.info("Initialized new model without pretraining")

    # Handle layer freezing for transfer learning
    if mae_pretrained and mae_train_layers > 0:
        freeze_base_layers(model.vit.encoder, mae_train_layers)
        if logger:
            logger.info(f"Froze all but last {mae_train_layers} encoder layers")

    return model

def freeze_base_layers(encoder, mae_train_layers):
    """
    Freeze specific layers of MAE encoder for transfer learning.

    Args:
        encoder (nn.Module): MAE encoder module
        mae_train_layers (int): Number of last layers to keep trainable
                              If -1, all layers remain trainable

    Notes:
        - Earlier layers are frozen first
        - Layer parameters are frozen by setting requires_grad=False
        - Only affects encoder layers, other model components remain trainable
    """
    if mae_train_layers > 0:
        num_layers = len(encoder.layer)
        layers_to_freeze = num_layers - mae_train_layers

        # Freeze earlier layers while keeping later layers trainable
        for i in range(layers_to_freeze):
            for param in encoder.layer[i].parameters():
                param.requires_grad = False

def setup_optimizer_and_scheduler(model, base_lr, batch_size, weight_decay, warmup_epochs, total_epochs, constant_base_lr=True):
    """
    Set up optimizer and learning rate scheduler for BERT model.
    Uses same warmup + cosine annealing schedule as before.
    """
    # Determine actual learning rate
    actual_lr = base_lr if constant_base_lr else base_lr * (batch_size / 256)

    # Initialize AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=actual_lr, weight_decay=weight_decay)

    # Setup warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # Setup cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs
    )

    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    return optimizer, scheduler

def unpatchify(x, patch_size=16):
    """
    Convert patches back to image format.

    Args:
        x (torch.Tensor): Input tensor of shape [B, num_patches, patch_size*patch_size*channels]
        patch_size (int): Size of each patch

    Returns:
        torch.Tensor: Reconstructed image of shape [B, channels, H, W]
    """
    batch_size = x.shape[0]
    num_patches = x.shape[1]
    h = w = int(num_patches ** 0.5)
    channels = 3

    # Reshape to [B, h, w, p, p, C]
    x = x.reshape(batch_size, h, w, patch_size, patch_size, channels)

    # Permute and reshape to image format [B, C, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, channels, h*patch_size, w*patch_size)
    return x

def denormalize(image):
    """Convert normalized tensor back to [0,1] range for visualization
       Mean and variance come from imagenet statistics.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(image.device)
    return image * std + mean

def train_channel(*,
                 model, optimizer, scheduler,
                 train_loader, dev_loader,
                 num_epochs, save_epoch_interval, result_dir, device,
                 pixel_variance=1.0, num_save_images=8, save_image_interval=5,
                 mtest_loader=None, overwrite=False):
    """
    Train MAE model using KL divergence loss while monitoring conditional entropy.

    Functional Load Calculation:
    FL = I(X;Y_clean) - I(X;Y_noisy)
    = [H(X) - H(X|Y_clean)] - [H(X) - H(X|Y_noisy)]
    = H(X|Y_noisy) - H(X|Y_clean)
    = -1/N Σ log P(X|Y_noisy) - (-1/N Σ log P(X|Y_clean))
    Where:
    - X: Original signal
    - Y_clean: Received through clean channel
    - Y_noisy: Received through contrast-distorted channel
    - N: number of pixels

    Two key calculations:
    Channel model: X->distorted channel->Y->restoration->Y'->X
    0) Assume points of an image
        Conditional dependence (upon the whole image)
        Each point ~ N(X,σ²) where X is true pixel value
    1) Optimization objective - KL divergence between Gaussians:
        KL(N(X,σ²) || N(Y',σ²)) = (X-Y')²/(2σ²)
        where:
        - X is true pixel value
        - Y' is predicted pixel value
        - σ² is pixel_variance

    2) Monitoring metric - Conditional entropy:
        H(X|Y) = -1/N * Σ log P(X|Y)
        where P(X|Y) is Gaussian with mean Y and variance σ²
        P(X|Y): Under the predicted distribution N(Y’,σ²), the probability of true value
        Direct formula: -log P(X|Y) = (X-Y')²/(2σ²) + log(2πσ²)/2
        where:
        - X is true pixel value
        - Y' is predicted pixel value, a function on distorted signal Y
        - σ² is pixel_variance
        - N: number of pixels
        # Note that differential conditional entropy can be negative
        # Common in continuous domains with good reconstruction
        # Can be negative when reconstruction precision exceeds reference scale
        # We can predict X from Y with better-than-unit precision
        # Our distribution is more concentrated than a uniform reference
        # It measures relative uncertainty; may focus on entropy differences rather than absolute values

    Args:
        model (ViTMAEForPreTraining): MAE model for image reconstruction
        optimizer (torch.optim.Optimizer): Model optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        train_loader (DataLoader): Training data loader
        dev_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs to train
        save_epoch_interval (int): Save model checkpoint every N epochs
        result_dir (str): Directory to save results and checkpoints
        device (torch.device): Device to train on (cpu or cuda)
        pixel_variance (float, optional): Variance for pixel distributions. Defaults to 1.0
        num_save_images (int, optional): Number of images to save in tensorboard. Defaults to 8
        save_image_interval (int, optional): Save images every N epochs. Defaults to 5
        mtest_loader (DataLoader, optional): Monitoring test set loader. Defaults to None
        overwrite (bool, optional): Whether to overwrite result directory. Defaults to False
    """
    # Setup logging and directories
    overwrite_result_directory(result_dir, overwrite)
    logger = init_logger(os.path.join(args.result_dir, "report.log"))
    logger.info(args)

    logger.info('Training MAE model for image reconstruction')

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    best_dev_loss = float('inf')
    best_dev_epoch = 0
    epoch = 0

    while epoch < num_epochs:
        # Training Phase
        model.train()
        train_kl_loss = 0.0  # Track KL divergence
        total_mse = 0.0      # Track total MSE
        total_pixels = 0     # Count of processed pixels
        info_table = []      # For metrics logging

        for images, _ in tqdm.tqdm(train_loader, ascii=True, ncols=50):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate MSE between predicted and true pixels
            reconstructed_images = unpatchify(outputs.logits, patch_size=16)  # or whatever patch_size you're using
            if reconstructed_images.shape != images.shape:
                raise ValueError(f"Shape mismatch: reconstructed {reconstructed_images.shape} vs original {images.shape}")
            mse_loss = F.mse_loss(reconstructed_images, images, reduction='sum')
            num_pixels = images.numel()

            # Calculate KL divergence for optimization: (X-Y)²/(2σ²)
            kl_loss = mse_loss/(2*pixel_variance)
            kl_per_pixel = kl_loss/num_pixels

            # Calculate differential conditional entropy
            # Negative conditional entropy indicates:
            # - Reconstruction precision better than reference scale
            # - Model achieves sub-unit accuracy in prediction
            # - Common in continuous domains with good reconstruction
            # - Conditional entropy is relative uncertainty
            # Calculate conditional entropy directly: -log P(X|Y) = (X-Y)²/(2σ²) + log(2πσ²)/2
            cond_entropy = mse_loss/(2*pixel_variance) + num_pixels*np.log(2*np.pi*pixel_variance)/2
            bits_per_pixel = cond_entropy / (num_pixels * np.log(2))
            mse_per_pixel = mse_loss / num_pixels

            # Optimize using KL divergence
            optimizer.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
            optimizer.step()

            # Accumulate statistics
            train_kl_loss += kl_per_pixel.item()
            total_mse += mse_loss.item()
            total_pixels += num_pixels

            # Check for NaN
            if torch.isnan(kl_loss):
                raise ValueError("NaN detected in loss calculation")

        # Calculate training metrics
        train_kl_loss /= len(train_loader)  # Average KL divergence per pixel
        train_mse_per_pixel = total_mse / total_pixels

        # Calculate conditional entropy rate using direct formula -log P(X|Y) = (X-Y)²/(2σ²) + log(2πσ²)/2
        train_cond_entropy = total_mse/(2*pixel_variance) + total_pixels*np.log(2*np.pi*pixel_variance)/2
        train_cond_entropy_rate = train_cond_entropy / (total_pixels * np.log(2))

        train_avg_received_per_pixel = 2**train_cond_entropy_rate
        train_avg_received_deviation = train_avg_received_per_pixel - 1.0

        # Record training metrics
        info_table.append([
            epoch,
            "train_set",
            f"{train_kl_loss:.6e}",        # KL divergence per pixel
            f"{train_mse_per_pixel:.6e}",  # MSE per pixel
            f"{train_cond_entropy_rate:.6e}",  # H(X|Y) in bits/pixel
            f"1{train_avg_received_deviation:+.6e}"  # Avg distortion
        ])

        # Log training metrics to tensorboard
        writer.add_scalar('KLDivergence/train', train_kl_loss, epoch)
        writer.add_scalar('MSE/train', train_mse_per_pixel, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits/train', train_cond_entropy_rate, epoch)
        writer.add_scalar('AvgReceivedPerPixelDeviation/train', train_avg_received_deviation, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Save training images
        if epoch % save_image_interval == 0:
            writer.add_images('Train/Original', denormalize(images[:num_save_images]), epoch)
            writer.add_images('Train/Reconstructed', denormalize(reconstructed_images[:num_save_images]), epoch)

        # Validation Phase
        model.eval()
        eval_kl_loss = 0.0
        total_mse = 0.0
        total_pixels = 0

        with torch.no_grad():
            for images, _ in tqdm.tqdm(dev_loader, ascii=True, ncols=50):
                images = images.to(device)
                outputs = model(images)

                # Calculate MSE
                reconstructed_images = unpatchify(outputs.logits, patch_size=16)
                if reconstructed_images.shape != images.shape:
                    raise ValueError(f"Shape mismatch in validation: reconstructed {reconstructed_images.shape} vs original {images.shape}")
                mse_loss = F.mse_loss(reconstructed_images, images, reduction='sum')
                num_pixels = images.numel()

                # Calculate KL divergence and conditional entropy
                kl_loss = mse_loss/(2*pixel_variance)
                kl_per_pixel = kl_loss/num_pixels

                # Calculate conditional entropy directly
                cond_entropy = mse_loss/(2*pixel_variance) + num_pixels*np.log(2*np.pi*pixel_variance)/2
                bits_per_pixel = cond_entropy / (num_pixels * np.log(2))

                if torch.isnan(bits_per_pixel):
                    raise ValueError("NaN detected in validation loss")

                # Accumulate statistics
                eval_kl_loss += kl_per_pixel.item()
                total_mse += mse_loss.item()
                total_pixels += num_pixels

        # Calculate validation metrics
        eval_kl_loss /= len(dev_loader)  # Average KL divergence per pixel
        dev_mse_per_pixel = total_mse / total_pixels

        # Calculate validation conditional entropy rate
        dev_cond_entropy = total_mse/(2*pixel_variance) + total_pixels*np.log(2*np.pi*pixel_variance)/2
        dev_cond_entropy_rate = dev_cond_entropy / (total_pixels * np.log(2))

        dev_avg_received_per_pixel = 2**dev_cond_entropy_rate
        dev_avg_received_deviation = dev_avg_received_per_pixel - 1.0

        # Record validation metrics
        info_table.append([
            epoch,
            "dev_set",
            f"{eval_kl_loss:.6e}",
            f"{dev_mse_per_pixel:.6e}",
            f"{dev_cond_entropy_rate:.6e}",
            f"1{dev_avg_received_deviation:+.6e}"
        ])

        # Log validation metrics
        writer.add_scalar('KLDivergence/dev', eval_kl_loss, epoch)
        writer.add_scalar('MSE/dev', dev_mse_per_pixel, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits/dev', dev_cond_entropy_rate, epoch)
        writer.add_scalar('AvgReceivedPerPixelDeviation/dev', dev_avg_received_deviation, epoch)

        # Save validation images
        if epoch % save_image_interval == 0:
            writer.add_images('Val/Original', denormalize(images[:num_save_images]), epoch)
            writer.add_images('Val/Reconstructed', denormalize(reconstructed_images[:num_save_images]), epoch)

        # Test Monitoring Phase
        if mtest_loader:
            model.eval()
            test_kl_loss = 0.0
            total_mse = 0.0
            total_pixels = 0

            with torch.no_grad():
                for images, _ in tqdm.tqdm(mtest_loader, ascii=True, ncols=50):
                    images = images.to(device)
                    outputs = model(images)

                    reconstructed_images = unpatchify(outputs.logits, patch_size=16)
                    if reconstructed_images.shape != images.shape:
                        raise ValueError(f"Shape mismatch in test: reconstructed {reconstructed_images.shape} vs original {images.shape}")
                    mse_loss = F.mse_loss(reconstructed_images, images, reduction='sum')
                    num_pixels = images.numel()

                    # Calculate KL divergence
                    kl_loss = mse_loss/(2*pixel_variance)
                    kl_per_pixel = kl_loss/num_pixels

                    # Calculate conditional entropy directly
                    cond_entropy = mse_loss/(2*pixel_variance) + num_pixels*np.log(2*np.pi*pixel_variance)/2
                    bits_per_pixel = cond_entropy / (num_pixels * np.log(2))

                    if torch.isnan(bits_per_pixel):
                        raise ValueError("NaN detected in test loss")

                    test_kl_loss += kl_per_pixel.item()
                    total_mse += mse_loss.item()
                    total_pixels += num_pixels

            # Calculate test metrics
            test_kl_loss /= len(mtest_loader)
            test_mse_per_pixel = total_mse / total_pixels

            # Calculate test conditional entropy rate
            test_cond_entropy = total_mse/(2*pixel_variance) + total_pixels*np.log(2*np.pi*pixel_variance)/2
            test_cond_entropy_rate = test_cond_entropy / (total_pixels * np.log(2))

            test_avg_received_per_pixel = 2**test_cond_entropy_rate
            test_avg_received_deviation = test_avg_received_per_pixel - 1.0

            # Record test metrics
            info_table.append([
                epoch,
                "test_set",
                f"{test_kl_loss:.6e}",
                f"{test_mse_per_pixel:.6e}",
                f"{test_cond_entropy_rate:.6e}",
                f"1{test_avg_received_deviation:+.6e}"
            ])

            # Log test metrics
            writer.add_scalar('KLDivergence/test', test_kl_loss, epoch)
            writer.add_scalar('MSE/test', test_mse_per_pixel, epoch)
            writer.add_scalar('ConditionalEntropyRate_bits/test', test_cond_entropy_rate, epoch)
            writer.add_scalar('AvgReceivedPerPixelDeviation/test', test_avg_received_deviation, epoch)

            # Save test images
            if epoch % save_image_interval == 0:
                writer.add_images('Test/Original', denormalize(images[:num_save_images]), epoch)
                writer.add_images('Test/Reconstructed', denormalize(reconstructed_images[:num_save_images]), epoch)

        # Print metrics table
        logger.info("\n" + tabulate.tabulate(
            info_table,
            headers=['epoch', 'dataset', 'kl_divergence', 'mse_per_pixel',
                    'cond_entropy_bits', 'avg_received_per_pixel'],
            floatfmt='.6e',
            tablefmt='rst'
        ))

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")

        # Save periodic checkpoints
        if epoch % save_epoch_interval == 0:
            checkpoint_path = os.path.join(result_dir, f"model_e{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Model saved: {checkpoint_path}")

        # Save best model based on KL divergence
        if eval_kl_loss < best_dev_loss:
            best_dev_loss = eval_kl_loss
            best_dev_epoch = epoch
            logger.info(f"New best dev KL loss {best_dev_loss:.3f} at epoch {best_dev_epoch}")
            best_model_path = os.path.join(result_dir, "model_best_dev.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, best_model_path)
            logger.info(f"Best model saved: {best_model_path}")

        # Save latest model
        latest_model_path = os.path.join(result_dir, "model.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, latest_model_path)
        logger.info(f"Latest model saved: {latest_model_path}")

        # Check for continuing training
        if epoch == num_epochs - 1 and not args.exit:
            command = "python " + ' '.join([x for x in sys.argv])
            message = f"command: '{command}'\nresult: '{result_dir}'\n"
            continue_or_not, add_epochs = continue_train(message)
            if continue_or_not and add_epochs:
                num_epochs += add_epochs
                logger.info(f"Adding {add_epochs} more epochs")

        epoch += 1

    writer.close()

# Main argument parser
parser = argparse.ArgumentParser(description='Train MAE model for image reconstruction conditional entropy')

# Data arguments
parser.add_argument('--train_dir', type=str,
                   default='/data/share/bin-wu/data/other/image/imagenet/sample_imagenet/test',
                   help='Directory containing training images')
parser.add_argument('--dev_dir', type=str,
                   default='/data/share/bin-wu/data/other/image/imagenet/sample_imagenet/val',
                   help='Directory containing validation images')
parser.add_argument('--mtest_dir', type=str,
                   default=None,
                   help='Directory containing monitoring test images')
parser.add_argument('--batch_size', type=int, default=128,
                   help='Training batch size')

# Model arguments
parser.add_argument('--image_size', type=int, default=224,
                   help='Image size (assumes square images)')
parser.add_argument('--patch_size', type=int, default=16,
                   help='Patch size for MAE')
parser.add_argument('--dim', type=int, default=768,
                   help='Hidden dimension size')
parser.add_argument('--depth', type=int, default=12,
                   help='Number of transformer layers')
parser.add_argument('--heads', type=int, default=12,
                   help='Number of attention heads')
parser.add_argument('--mlp_dim', type=int, default=3072,
                   help='Dimension of MLP layer')
parser.add_argument('--dropout', type=float, default=0.1,
                   help='Dropout rate')
parser.add_argument('--mae_pretrained', type=str, default="",
                   help='Path to pretrained MAE model, "base" for facebook/vit-mae-base, or "" for no pretraining')
parser.add_argument('--mae_train_layers', type=int, default=-1,
                   help='Number of last layers to train in MAE when using pretrained model. -1 means train all layers')
parser.add_argument('--pixel_variance', type=float, default=0.05,
                   help='Assumed variance for pixel distributions (default: 0.05 close to ImageNet variance)')
parser.add_argument('--num_save_images', type=int, default=8,
                   help='Number of images to save for visualization')

# Optimizer arguments
parser.add_argument('--base_lr', type=float, default=1.5e-4,
                   help='Base learning rate')
parser.add_argument('--weight_decay', type=float, default=0.05,
                   help='Weight decay for AdamW')
parser.add_argument('--warmup_epochs', type=int, default=10,
                   help='Number of warmup epochs')
parser.add_argument('--total_epochs', type=int, default=100,
                   help='Total epochs for scheduler cycle')

# Training arguments
parser.add_argument('--num_epochs', type=int, default=3, # 100,
                   help='Number of training epochs')
parser.add_argument('--save_epoch_interval', type=int, default=10,
                   help='Save model checkpoint every N epochs')
parser.add_argument('--save_image_interval', type=int, default=5,
                   help='Save image samples every N epochs')
parser.add_argument('--result_dir', type=str, default='exp/fl_ce_imagenet/test/sample_imagenet',
                   help='Directory to save results')

# Other arguments
parser.add_argument('--seed', type=int, default=2025,
                   help='Random seed for reproducibility')
parser.add_argument('--gpu', type=str, default='auto',
                   help='GPU selection ("auto" or specific id)')
parser.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing result directory')
parser.add_argument('--exit', action='store_true',
                   help='Exit after training without prompting to continue')

args = parser.parse_args()

set_seed(args.seed)
device = set_device(args.gpu)

# Create datasets and dataloaders
train_dataset = ImageNetDataset(args.train_dir)
dev_dataset = ImageNetDataset(args.dev_dir)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

mtest_loader = None
if args.mtest_dir:
    mtest_dataset = ImageNetDataset(args.mtest_dir)
    mtest_loader = DataLoader(mtest_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

# Create model
model = create_mae_model(
    image_size=args.image_size,
    patch_size=args.patch_size,
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    channels=3,  # Always 3 for ImageNet
    mae_pretrained=args.mae_pretrained,
    mae_train_layers=args.mae_train_layers
)
model = model.to(device)

# Setup optimizer and scheduler
optimizer, scheduler = setup_optimizer_and_scheduler(
    model=model,
    base_lr=args.base_lr,
    batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_epochs=args.warmup_epochs,
    total_epochs=args.total_epochs
)

# Train model with explicit parameters
train_channel(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    dev_loader=dev_loader,
    num_epochs=args.num_epochs,
    save_epoch_interval=args.save_epoch_interval,
    result_dir=args.result_dir,
    device=device,
    pixel_variance=args.pixel_variance,
    num_save_images=args.num_save_images,
    save_image_interval=args.save_image_interval,
    mtest_loader=mtest_loader,
    overwrite=args.overwrite
)
