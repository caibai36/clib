#!/usr/bin/env python3
"""
Speaker Identification Trainer and Evaluator using Vision Transformer with MAE

This script trains and evaluates a speaker identification model using the pretrained MAE-ViT architecture.
It uses spectrograms as input features and identifies speakers from the dataset.

Default datasets:
- Training, development, and test sets from exp/caller_identification/data_division/

Default features:
- exp/caller_identification/data_division/train_features.npy
- exp/caller_identification/data_division/dev_features.npy
- exp/caller_identification/data_division/test_features.npy

Default speaker IDs:
- exp/caller_identification/data_division/train_speaker_ids.npy
- exp/caller_identification/data_division/dev_speaker_ids.npy
- exp/caller_identification/data_division/test_speaker_ids.npy

Default speaker-to-ID mapping:
- exp/caller_identification/data_division/spk2id.yaml

Example usage for training:
    python local/caller_identification_run.py

Example usage for evaluation:
    python local/caller_identification_run.py \
        --eval_feature exp/data/division_nas5_b2_mae_pretrained_all_days/dev_input.npy \
        --eval_config exp/data/division_nas5_b2_mae_pretrained_all_days/dev_config.csv \
        --eval_output_config exp/caller_identification_exp/eval/config_dev_day153.csv \
        --eval_model exp/caller_identification_exp/train/model.ckpt
"""

import os
import sys
import datetime
import logging
import glob
import shutil
import yaml
import csv

import math
import random
import argparse

import GPUtil
import tqdm
import tabulate

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

from torchvision import transforms

# Use ViTMAE ViT encoder
from transformers import ViTMAEForPreTraining

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    """
    Set the seed for reproducibility across NumPy and PyTorch.
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
    """ Initialize a logger to terminal and file at the same time. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", "%d/%m/%Y %H:%M:%S")

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
        message (str): Optional message to display before the prompt. Default is an empty string.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the user decides to continue training, False otherwise.
            - int: The number of additional epochs to train, as specified by the user.
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

class MAEViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder using MAE's architecture without masking mechanism for single-channel spectrogram input.

    This model:
    - Uses the encoder part of MAE architecture without any masking
    - Handles single-channel to 3-channel conversion for spectrograms
    - Supports both training from scratch and loading pretrained weights
    - Provides flexible pooling options and latent representation extraction
    - Uses ImageNet normalization for pretrained model compatibility

    Args:
        image_size (int): Input image size (assumes square images). Default: 224 (MAE default)
        patch_size (int): Size of patches. Default: 16 (MAE default)
        num_classes (int): Number of output classes
        dim (int): Hidden dimension size. Default: 768 (MAE default)
        depth (int): Number of transformer layers. Default: 12 (MAE default)
        heads (int): Number of attention heads. Default: 12 (MAE default)
        mlp_dim (int): Dimension of MLP layer. Default: 3072 (MAE default)
        pool (str): Pooling type ('cls' or 'mean'). Default: 'cls'
        channels (int): Number of input channels. Default: 1 (for spectrograms)
        dim_head (int): Dimension of each attention head. Default: 64
        dropout (float): Dropout rate. Default: 0.0
        emb_dropout (float): Embedding dropout rate. Default: 0.0
        return_attention (bool): Whether to return attention weights. Default: False
        return_logits (bool): Whether to return logits instead of probabilities. Default: False
        return_latent (bool): Whether to return latent representations. Default: False
        pretrained (bool): Whether to load pretrained MAE weights. Default: False

    Input shape:
        - Single channel: (batch_size, 1, height, width) or (batch_size, height, width)
        - Will be converted to: (batch_size, 3, 224, 224)

    Output shape:
        Based on configuration:
        - If return_latent: (batch_size, dim)
        - If return_logits: (batch_size, num_classes) before softmax
        - Otherwise: (batch_size, num_classes) after softmax
    """
    def __init__(self, *, image_size=224, patch_size=16, num_classes, dim=768,
                depth=12, heads=12, mlp_dim=3072, pool='cls', channels=1,
                dim_head=64, dropout=0., emb_dropout=0., return_attention=False,
                return_logits=False, return_latent=False, mae_pretrained='base',
                mae_train_layers=-1):
        super().__init__()

        # Initialize ViT model
        if mae_pretrained:
            if mae_pretrained == 'base':
                # Get the ViT encoder from pretrained MAE base model
                mae = ViTMAEForPreTraining.from_pretrained(
                    'facebook/vit-mae-base',
                    mask_ratio=0.0  # Disable masking
                )
                self.vit = mae.vit
            else:
                # Load from checkpoint saved by riken_mae_pretrained.py
                checkpoint = torch.load(mae_pretrained, map_location='cpu')
                # Create a new MAE model first
                mae = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base', mask_ratio=0.0)
                # Load the saved state dict
                mae.load_state_dict(checkpoint['model'])
                self.vit = mae.vit

            # Freeze layers based on mae_train_layers parameter
            self.freeze_base_layers(mae_train_layers)
        else:
            # Initialize fresh MAE with masking disabled
            from transformers import ViTMAEConfig, ViTMAEModel

            config = ViTMAEConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                intermediate_size=mlp_dim,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                num_channels=3,  # Always use 3 channels after input processing
                mask_ratio=0.0  # Disable masking
            )
            self.vit = ViTMAEModel(config)

        # Classification head for supervised learning
        self.mlp_head = nn.Linear(dim, num_classes)

        # Model configuration
        self.pool = pool
        self.return_attention = return_attention
        self.return_logits = return_logits
        self.return_latent = return_latent

        # Input processing configuration
        self.patch_size = patch_size
        self.image_size = image_size

        # Input transformation pipeline
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )

    def freeze_base_layers(self, mae_train_layers):
        """
        Freeze specific layers based on mae_train_layers parameter when using pretrained model.
        All parameters are trainable by default, only freezing layers if specified.

        Args:
            mae_train_layers (int): Number of last layers to keep trainable.
                                If -1, all layers remain trainable.
        """
        if mae_train_layers > 0:
            # Calculate which layers to freeze
            num_layers = len(self.vit.encoder.layer)
            layers_to_freeze = num_layers - mae_train_layers

            # Freeze only the specified number of early layers
            for i in range(layers_to_freeze):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = False

    def process_input(self, x):
        """
        Process single-channel spectrogram input to match MAE's expected input format.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
                            or (batch_size, height, width)

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, 3, 224, 224)

        Steps:
            1. Add channel dimension if needed
            2. Resize to required dimensions
            3. Normalize to [0, 1]
            4. Convert to 3 channels
            5. Apply ImageNet normalization
        """
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Resize if necessary
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = transforms.Resize((self.image_size, self.image_size))(x)

        # Normalize to [0, 1]
        x = (x - x.min()) / (x.max() - x.min())

        # Convert to 3 channels
        x = x.repeat(1, 3, 1, 1)

        # Apply ImageNet normalization
        x = self.normalize(x)

        return x

    def forward(self, img, y=None):
        """
        Forward pass of the model.

        Args:
            img (torch.Tensor): Input spectrogram
            y (torch.Tensor, optional): Ground truth labels for supervised training

        Returns:
            Different return formats based on configuration:
            - If return_latent:
                tuple: (latent_features, attention_weights)
            - If y is provided:
                tuple: (output, loss, accuracy, attention_weights)
            - Otherwise:
                tuple: (output, attention_weights)

            Where:
            - latent_features: tensor of shape (batch_size, dim)
            - output: tensor of shape (batch_size, num_classes)
            - attention_weights: list of attention matrices if return_attention=True
            - loss: scalar tensor if y is provided
            - accuracy: list of per-sample accuracy if y is provided
        """
        # Process input to match expected format
        img = self.process_input(img)

        # Forward pass through ViT encoder (no masking)
        outputs = self.vit(
            img,
            output_attentions=self.return_attention,
            return_dict=True
        )

        # Get latent features based on pooling strategy
        if self.pool == 'cls':
            latent = outputs.last_hidden_state[:, 0]  # Use CLS token
        else:
            latent = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Get attention weights if requested
        attention_weights = outputs.attentions if self.return_attention else None

        # Return latent features if requested
        if self.return_latent:
            return latent, attention_weights

        # Classification head
        logits = self.mlp_head(latent)
        output = logits if self.return_logits else F.softmax(logits, dim=1)

        # Handle supervised training case
        if y is not None:
            y = y.type(torch.LongTensor).to(y.device)
            loss = F.cross_entropy(logits, y)
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == y).float().tolist()
            return output, loss, accuracy, attention_weights

        return output, attention_weights

class SpeakerIDDataset(Dataset):
    """
    A PyTorch Dataset class for speaker identification data.

    Args:
        features (numpy.ndarray): Input features, of shape (data_length, 257, 256).
        speaker_ids (numpy.ndarray): Speaker IDs corresponding to features.
        train (bool): Whether the dataset is used for training or evaluation.
        apply_random_shift (bool): Whether to apply data augmentation of random shifts.
    """
    def __init__(self, features, speaker_ids, train=True, apply_random_shift=True):
        self.features = features
        self.speaker_ids = speaker_ids
        self.train = train
        self.apply_random_shift = apply_random_shift

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        speaker_id = self.speaker_ids[idx]

        # Data augmentation for training
        if self.train and self.apply_random_shift:
            ver_shift = random.randint(-5, 5)
            hor_shift = random.randint(-5, 5)
            x = np.roll(x, (ver_shift, hor_shift), axis=(0, 1))

        return x.astype(np.float32), speaker_id.astype(np.int64)

def create_dataloader(features_path, speaker_ids_path, batch_size, train=True):
    """
    Create a data loader for speaker identification.

    Args:
        features_path (str): Path to the features file.
        speaker_ids_path (str): Path to the speaker IDs file.
        batch_size (int): Batch size for the data loader.
        train (bool): Whether to create a data loader for training or evaluation.

    Returns:
        torch.utils.data.DataLoader: The created data loader.
    """
    # Load data from input files
    features = np.load(features_path)
    speaker_ids = np.load(speaker_ids_path)

    # Create dataset instance
    dataset = SpeakerIDDataset(features, speaker_ids, train=train, apply_random_shift=train)

    # Create DataLoader with the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_loader

def setup_optimizer_and_scheduler(model, base_lr, batch_size, weight_decay, warmup_epochs, total_epochs, constant_base_lr=True):
    """
    Set up the optimizer and learning rate scheduler for the model.

    Args:
        model (nn.Module): The PyTorch model to optimize.
        base_lr (float): Base learning rate.
        batch_size (int): Batch size for training.
        weight_decay (float): Weight decay factor for L2 regularization.
        warmup_epochs (int): Number of epochs for the warmup phase.
        total_epochs (int): Total number of epochs including warmup and cosine annealing phases.
        constant_base_lr (bool): If True, use the base_lr as is; if False, scale the learning rate based on batch size.

    Returns:
        tuple: (optimizer, scheduler)
    """
    # Determine the actual learning rate
    if constant_base_lr:
        actual_lr = base_lr
    else:
        actual_lr = base_lr * (batch_size / 256)  # Scale based on batch size

    # Initialize AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=actual_lr, weight_decay=weight_decay)

    # Define warmup scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    # Define cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

    # Combine warmup and cosine annealing schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    return optimizer, scheduler

def train(model, optimizer, scheduler, train_loader, dev_loader, num_epochs, save_epoch_interval, result_dir, mtest_loader=None):
    """
    Function for training the speaker identification model.

    Args:
        model: The model to be trained.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        train_loader: Data loader for the training set.
        dev_loader: Data loader for the development set.
        num_epochs: Number of training epochs.
        save_epoch_interval: Interval for saving the model.
        result_dir: Path to save the trained models.
        mtest_loader: Data loader for the test set (for monitoring).
    """
    # Whether to overwrite the result directory when it exists
    overwrite_result_directory(result_dir, args.overwrite)

    logger = init_logger(os.path.join(result_dir, "report.log"))
    logger.info(args)
    logger.info('Training speaker identification model')
    if args.mae_pretrained:
        logger.info(f'Using pretrained MAE model: {args.mae_pretrained}')

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    best_dev_loss = float('inf')
    best_dev_epoch = 0
    epoch = 0

    while epoch < num_epochs:
        accuracies = []
        train_loss = 0.0
        info_table = []
        model.train()  # Set the model to training mode

        # Iterate over the training data loader
        for batch_features, batch_speaker_ids in tqdm.tqdm(train_loader, ascii=True, ncols=50):
            # Move the batch data to the device
            batch_features = batch_features.to(device)
            batch_speaker_ids = batch_speaker_ids.to(device)

            # Forward pass through the model
            _, loss, accurs, _ = model(batch_features, batch_speaker_ids)
            accuracies.extend([1 if val else 0 for val in accurs])
            train_loss += loss.item()

            # Backward pass and optimization step
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
            optimizer.step()

            # Check for NaN values in the loss
            if np.isnan(loss.item()):
                raise ValueError("NaN detected in loss")

        # Compute average training loss and accuracy
        train_loss /= len(train_loader)
        train_accuracy = np.mean(accuracies)

        # Log the training metrics
        info_table.append([epoch, "train_set", train_loss, train_accuracy])
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Evaluate on the development set
        model.eval()
        accuracies = []
        eval_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_speaker_ids in tqdm.tqdm(dev_loader, ascii=True, ncols=50):
                batch_features = batch_features.to(device)
                batch_speaker_ids = batch_speaker_ids.to(device)

                _, loss, accurs, _ = model(batch_features, batch_speaker_ids)
                accuracies.extend([1 if val else 0 for val in accurs])
                eval_loss += loss.item()

        # Compute average evaluation loss and accuracy
        eval_loss /= len(dev_loader)
        eval_accuracy = np.mean(accuracies)

        # Test set monitoring (if provided)
        if mtest_loader:
            model.eval()
            accuracies = []
            mtest_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_speaker_ids in tqdm.tqdm(mtest_loader, ascii=True, ncols=50):
                    batch_features = batch_features.to(device)
                    batch_speaker_ids = batch_speaker_ids.to(device)

                    _, loss, accurs, _ = model(batch_features, batch_speaker_ids)
                    accuracies.extend([1 if val else 0 for val in accurs])
                    mtest_loss += loss.item()

            mtest_loss /= len(mtest_loader)
            mtest_accuracy = np.mean(accuracies)
            info_table.append([epoch, "test_set", mtest_loss, mtest_accuracy])
            writer.add_scalar('Loss/test', mtest_loss, epoch)
            writer.add_scalar('Accuracy/test', mtest_accuracy, epoch)

        # Log evaluation metrics
        info_table.append([epoch, "dev_set", eval_loss, eval_accuracy])
        logger.info("\n" + tabulate.tabulate(info_table, headers=['epoch', 'dataset', 'loss', 'acc'], floatfmt='.4f', tablefmt='rst'))

        writer.add_scalar('Loss/dev', eval_loss, epoch)
        writer.add_scalar('Accuracy/dev', eval_accuracy, epoch)

        # Step the learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")

        # Save model checkpoints
        if epoch % save_epoch_interval == 0:
            model_path = os.path.join(result_dir, f"model_e{epoch}.ckpt")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved: {model_path}")

        # Save the best model based on development loss
        if eval_loss < best_dev_loss:
            best_dev_loss = eval_loss
            best_dev_epoch = epoch
            logger.info(f"Got a better dev loss {best_dev_loss:.4f} at epoch {best_dev_epoch} ... saving the model")
            model_path = os.path.join(result_dir, "model_best_dev.ckpt")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Best model saved: {model_path}")

        # Save the latest model
        model_path = os.path.join(result_dir, "model.ckpt")
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Latest model saved: {model_path}")

        # Continue training option
        if epoch == num_epochs - 1 and not args.exit:
            command = "python " + ' '.join([x for x in sys.argv])
            message = f"command: '{command}'\nresult: '{result_dir}'\n"
            continue_or_not, add_epochs = continue_train(message)
            if continue_or_not and add_epochs:
                num_epochs += add_epochs
                logger.info(f"Adding {add_epochs} more epochs")

        epoch += 1

    # Close the TensorBoard writer
    writer.close()

def evaluate_model(model, eval_feature_path, id2spk_map, eval_config_path=None, eval_output_config_path=None, batch_size=32):
    """
    Evaluate a trained speaker identification model on a given dataset.

    Args:
        model: The trained model to evaluate
        eval_feature_path (str): Path to the feature file to evaluate
        id2spk_map (dict): Mapping from speaker IDs to speaker names
        eval_config_path (str, optional): Path to the configuration file
        eval_output_config_path (str, optional): Path to save the output configuration with speaker predictions
        batch_size (int): Batch size for evaluation

    Returns:
        tuple: (predictions, accuracy) where predictions is a list of predicted speaker labels
               and accuracy is the overall accuracy (if speaker IDs are available)
    """
    logger = init_logger()
    logger.info(f"Evaluating model on {eval_feature_path}")

    # Load features
    features = np.load(eval_feature_path)
    logger.info(f"Loaded features with shape: {features.shape}")

    # Load config if provided
    config_df = None
    if eval_config_path and os.path.exists(eval_config_path):
        config_df = pd.read_csv(eval_config_path)
        logger.info(f"Loaded configuration with {len(config_df)} entries")

    # Set model to evaluation mode
    model.eval()

    # Process features in batches
    predictions = []
    confidences = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)

            # Get predictions
            outputs, _ = model(batch_features_tensor)

            # Convert to numpy and get predictions
            outputs_np = outputs.cpu().numpy()
            batch_predictions = outputs_np.argmax(axis=1)
            batch_confidences = outputs_np.max(axis=1)

            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)

            if (i + batch_size) % (5 * batch_size) == 0 or (i + batch_size) >= len(features):
                logger.info(f"Processed {i + len(batch_features)}/{len(features)} samples")

    # Convert numeric predictions to speaker names
    speaker_predictions = [id2spk_map[int(pred)] for pred in predictions]

    # Create output configuration
    if eval_output_config_path:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(eval_output_config_path), exist_ok=True)

        if config_df is not None:
            # Add speaker predictions to existing config
            config_df['predicted_speaker'] = speaker_predictions
            config_df['confidence'] = confidences
            config_df.to_csv(eval_output_config_path, index=False)
        else:
            # Create a new config with just speaker predictions
            with open(eval_output_config_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['predicted_speaker', 'confidence'])
                for speaker, conf in zip(speaker_predictions, confidences):
                    writer.writerow([speaker, conf])

        logger.info(f"Saved predictions to {eval_output_config_path}")

    return speaker_predictions, confidences

def parse_args():
    """Parse command line arguments."""
    data_dir = "exp/caller_identification/data_division"

    parser = argparse.ArgumentParser(description="""
    Train and evaluate a speaker identification model using MAE-ViT.

    Example usage for training:
        python local/caller_identification_run.py

    Example usage for evaluation:
    python local/caller_identification_run.py --eval_feature exp/data/division_nas5_b2_mae_pretrained_all_days/dev_input.npy --eval_config exp/data/division_nas5_b2_mae_pretrained_all_days/dev_config.csv --eval_output_config exp/caller_identification_exp/eval/config_dev_day153.csv --eval_model exp/caller_identification_exp/train/model.ckpt
    """)

    # Data arguments
    parser.add_argument("--train_input", type=str, default=f"{data_dir}/train_features.npy",
                        help="Path to training features")
    parser.add_argument("--train_target", type=str, default=f"{data_dir}/train_speaker_ids.npy",
                        help="Path to training speaker IDs")
    parser.add_argument("--dev_input", type=str, default=f"{data_dir}/dev_features.npy",
                        help="Path to development features")
    parser.add_argument("--dev_target", type=str, default=f"{data_dir}/dev_speaker_ids.npy",
                        help="Path to development speaker IDs")
    parser.add_argument("--test_input", type=str, default=f"{data_dir}/test_features.npy",
                        help="Path to test features")
    parser.add_argument("--test_target", type=str, default=f"{data_dir}/test_speaker_ids.npy",
                        help="Path to test speaker IDs")
    parser.add_argument("--spk2id_yaml", type=str, default=f"{data_dir}/spk2id.yaml",
                        help="Path to speaker-to-ID mapping YAML file")
    parser.add_argument("--id2spk_yaml", type=str, default=f"{data_dir}/id2spk.yaml",
                        help="Path to ID-to-speaker mapping YAML file")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training and evaluation")

    # Model arguments
    parser.add_argument("--dim", type=int, default=768,
                        help="Model embedding dimension")
    parser.add_argument("--depth", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=3072,
                        help="Dimension of the MLP layer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--mae_pretrained", type=str, default="exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt",
                        help="Path to pretrained MAE model")
    parser.add_argument("--mae_train_layers", type=int, default=6,
                        help="Number of layers to fine-tune in the pretrained model")

    # Training arguments
    parser.add_argument("--base_lr", type=float, default=0.0001,
                        help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.3,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--save_epoch_interval", type=int, default=1,
                        help="Save model checkpoint every N epochs")

    # Evaluation arguments
    parser.add_argument("--eval_model", type=str, default=None,
                        help="Path to model checkpoint for evaluation (e.g., exp/caller_identification_exp/train/model.ckpt)")
    parser.add_argument("--eval_feature", type=str, default=None,
                        help="Path to feature file for evaluation (e.g., exp/data/division_nas5_b2_mae_pretrained_all_days/dev_input.npy)")
    parser.add_argument("--eval_config", type=str, default=None,
                        help="Path to configuration file for evaluation (e.g., exp/data/division_nas5_b2_mae_pretrained_all_days/dev_config.csv)")
    parser.add_argument("--eval_output_config", type=str, default=None,
                        help="Path to save output configuration with speaker predictions (e.g., exp/caller_identification_exp/eval/config_dev_day153.csv)")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu", type=str, default="1",
                        help="GPU device to use (e.g., '0', '1', or 'auto')")
    parser.add_argument("--result", type=str, default="exp/caller_identification_exp/train/",
                        help="Result directory for saving model checkpoints")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing result directory")
    parser.add_argument("--exit", action="store_true",
                        help="Exit after training without prompting for continuation")

    return parser.parse_args()

def main():
    """Main function to train or evaluate a speaker identification model."""
    global args, device
    args = parse_args()

    # Set up reproducibility and device
    set_seed(args.seed)
    device = set_device(args.gpu)
    print(f"Using device: {device}")

    # Check if we're in evaluation mode
    eval_mode = args.eval_feature is not None and args.eval_model is not None

    # Load speaker mapping
    with open(args.id2spk_yaml, 'r') as f:
        id2spk_map = yaml.safe_load(f)

    with open(args.spk2id_yaml, 'r') as f:
        spk2id_map = yaml.safe_load(f)

    num_speakers = len(spk2id_map)
    print(f"Number of speakers to identify: {num_speakers}")

    # Create model
    model = MAEViTEncoder(
        num_classes=num_speakers,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        mae_pretrained=args.mae_pretrained if not eval_mode else None,
        mae_train_layers=args.mae_train_layers
    ).to(device)

    # Load model weights for evaluation
    if eval_mode:
        print(f"Loading model from {args.eval_model} for evaluation...")
        checkpoint = torch.load(args.eval_model, map_location=device)
        model.load_state_dict(checkpoint["model"])

        # Perform evaluation
        print(f"Evaluating model on {args.eval_feature}...")
        evaluate_model(
            model=model,
            eval_feature_path=args.eval_feature,
            id2spk_map=id2spk_map,
            eval_config_path=args.eval_config,
            eval_output_config_path=args.eval_output_config,
            batch_size=args.batch_size
        )
        print("Evaluation completed!")
        return

    # Training mode
    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model=model,
        base_lr=args.base_lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs
    )

    # Create data loaders for training
    train_loader = create_dataloader(
        args.train_input,
        args.train_target,
        args.batch_size,
        train=True
    )

    dev_loader = create_dataloader(
        args.dev_input,
        args.dev_target,
        args.batch_size,
        train=False
    )

    # Create test loader for monitoring if test data is provided
    test_loader = create_dataloader(
        args.test_input,
        args.test_target,
        args.batch_size,
        train=False
    ) if args.test_input and args.test_target else None

    # Record start time
    start_time = datetime.datetime.now()

    # Train the model
    train(
        model,
        optimizer,
        scheduler,
        train_loader,
        dev_loader,
        args.num_epochs,
        args.save_epoch_interval,
        args.result,
        test_loader
    )

    # Calculate and display training duration
    duration = datetime.datetime.now() - start_time
    print(f'Training completed in {duration.seconds // 3600:02}:{(duration.seconds // 60) % 60:02}:{duration.seconds % 60:02}')

if __name__ == "__main__":
    main()
