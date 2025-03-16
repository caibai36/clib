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
import csv

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

def color_transform_standard(img_array, color_from, color_to, use_hsv_thresholds=True,
                           use_mask_only=False, no_color=False, no_value=False, bidirectional=False):
    """
    Transform colors in an image using HSV color space with multiple transformation options.

    This function provides several color transformation capabilities:
    1. Standard color transform: Convert one color to another (unidirectional or bidirectional)
    2. Mask only: Convert specified colors to grayscale
    3. No color: Convert entire image to grayscale
    4. No value: Replace all detected colors with their standard values without intensity

    The function works by:
    1. Converting RGB to HSV color space
    2. Detecting colors using predefined HSV ranges
    3. Applying the requested transformation:
       - Unidirectional: Transform source color to target color
       - Bidirectional: Simultaneously swap two colors
       - Mask: Convert specified colors to grayscale
       - No color: Full grayscale conversion
       - No value: Use standard colors without intensity

    Args:
        img_array (numpy.ndarray): Input RGB image array (H,W,3)
        color_from (str): First color in the pair
        color_to (str): Second color in the pair
        use_hsv_thresholds (bool, optional): Use saturation/value thresholds. Default True
        use_mask_only (bool, optional): Convert specified colors to grayscale. Default False
        no_color (bool, optional): Convert entire image to grayscale. Default False
        no_value (bool, optional): Use standard colors without intensity. Default False
        bidirectional (bool, optional): Swap colors instead of one-way transform. Default False

    Returns:
        numpy.ndarray: Transformed image as uint8 array

    Notes:
        - In bidirectional mode, colors are swapped simultaneously
        - HSV thresholds use standard values: saturation > 0.27, value > 0.2
        - Grayscale conversion uses standard weights: R:0.2989, G:0.5870, B:0.1140
    """
    # Convert to float for calculations
    img_float = img_array.astype(float)
    r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]

    # Handle full grayscale conversion
    if no_color:
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
        result = np.stack([grayscale, grayscale, grayscale], axis=2)
        return np.clip(result, 0, 255).astype(np.uint8)

    # Calculate HSV components
    max_rgb = np.maximum.reduce([r, g, b])
    min_rgb = np.minimum.reduce([r, g, b])
    diff = np.maximum(max_rgb - min_rgb, 1e-7)  # Avoid division by zero

    # Calculate hue for each channel maximum
    hue = np.zeros_like(r)

    # Red is maximum
    mask = (max_rgb == r)
    hue[mask] = 60 * (g[mask] - b[mask]) / diff[mask]

    # Green is maximum
    mask = (max_rgb == g)
    hue[mask] = 120 + 60 * (b[mask] - r[mask]) / diff[mask]

    # Blue is maximum
    mask = (max_rgb == b)
    hue[mask] = 240 + 60 * (r[mask] - g[mask]) / diff[mask]

    hue = (hue + 360) % 360  # Normalize to [0, 360]

    # Calculate saturation and value
    saturation = np.divide(diff, max_rgb, out=np.zeros_like(diff), where=max_rgb!=0)
    value = max_rgb / 255.0

    # Define standard color ranges in HSV
    color_ranges = {
        'red': [(0, 20), (340, 360)],  # Red wraps around 360/0
        'orange': (20, 45),
        'yellow': (45, 70),
        'green': (70, 165),
        'blue': (165, 260),
        'violet': (260, 340)
    }

    # Standard RGB values for colors
    color_values = {
        'red': (255, 0, 0),
        'orange': (255, 165, 0),
        'yellow': (255, 255, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'violet': (148, 0, 211)
    }

    # Handle no value mode - replace all detected colors with standard values
    if no_value:
        result = img_float.copy()
        # Initialize with middle gray (127) for non-colored regions
        for i in range(3):
            result[:,:,i] = 127.0

        # Process each standard color
        for color_name, ranges in color_ranges.items():
            if color_name == 'red':
                range1, range2 = ranges
                color_mask = ((hue >= range1[0]) & (hue <= range1[1])) | \
                            ((hue >= range2[0]) & (hue <= range2[1]))
            else:
                start, end = ranges
                color_mask = (hue >= start) & (hue <= end)

            if use_hsv_thresholds:
                color_mask = color_mask & (saturation > 0.27) & (value > 0.2)

            standard_color = np.array(color_values[color_name])
            for i in range(3):
                result[:,:,i][color_mask] = standard_color[i]

        return np.clip(result, 0, 255).astype(np.uint8)

    # Initialize result array
    result = img_float.copy()

    # Create masks for color detection
    def get_color_mask(color):
        if color == 'red':
            range1, range2 = color_ranges[color]
            mask = ((hue >= range1[0]) & (hue <= range1[1])) | \
                   ((hue >= range2[0]) & (hue <= range2[1]))
        else:
            start, end = color_ranges[color]
            mask = (hue >= start) & (hue <= end)

        if use_hsv_thresholds:
            mask = mask & (saturation > 0.27) & (value > 0.2)
        return mask

    # Get masks for both colors
    mask_from = get_color_mask(color_from)
    mask_to = get_color_mask(color_to)  # Always get both masks

    if use_mask_only:
        # Convert both colors to grayscale for both uni and bidirectional
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
        combined_mask = mask_from | mask_to  # Combine masks for both colors
        for i in range(3):
            result[:,:,i][combined_mask] = grayscale[combined_mask]
    else:
        # Handle color transformations
        if bidirectional:
            # Store original intensities
            intensity_from = max_rgb[mask_from] / 255
            intensity_to = max_rgb[mask_to] / 255

            # Get target colors
            color1 = np.array(color_values[color_to])
            color2 = np.array(color_values[color_from])

            # Apply bidirectional transformation
            for i in range(3):
                result[:,:,i][mask_from] = color1[i] * intensity_from
                result[:,:,i][mask_to] = color2[i] * intensity_to
        else:
            # Standard unidirectional transformation
            target_color = np.array(color_values[color_to])
            intensity_scale = max_rgb[mask_from] / 255
            for i in range(3):
                result[:,:,i][mask_from] = target_color[i] * intensity_scale

    return np.clip(result, 0, 255).astype(np.uint8)

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

class ImageNetContrastDataset(Dataset):
    """
    Dataset for computing functional load of color contrasts in ImageNet images.

    Provides original images for target (clean channel) and
    arrays for source (contrast-merged channel) creation in collate functions.

    Args:
        root_dir (str): Directory containing ImageNet images
        transform (callable, optional): Transform to be applied on images.
                                     Defaults to standard ImageNet normalization.

    Returns:
        dict: Contains:
            - image: Transformed tensor of original image [C,H,W]
            - array: Original image array for contrast merging
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Add base transforms for source array
        self.base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

        # Get and sort image files
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Apply base transforms to get consistent size before getting array
        base_image = self.base_transform(image)
        array = np.array(base_image)

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,  # Target (clean)
            'array': array   # For source (noisy) creation
        }

def train_collate_fn(batch, contrast_type=None, color_from=None, color_to=None, merge_type='mask'):
    """
    Collate function that handles contrast merging during training.
    Performs random sampling for uniform merge type at batch level.
    When contrast_type is None, behaves exactly like fl_ce_color.py,
    returning original images without any contrast merging.

    Args:
        batch: List of dictionaries from dataset
        contrast_type (str): Type of contrast to compute:
            - 'pair': Analyze specific color pair
            - 'all': Full grayscale conversion
            - 'value': Standard colors without intensity
        color_from (str): Source color for pair contrast
        color_to (str): Target color for pair contrast
        merge_type (str): How to merge colors in pair contrast:
            - 'uniform': Random sampling between colors
            - 'mask': Convert to grayscale

    Returns:
        dict: Contains:
            - clean: Original images [B,C,H,W] (target)
            - noisy: Contrast-merged images [B,C,H,W] (source)
    """
    # Stack target (clean) images
    target_images = torch.stack([item['image'] for item in batch])
    arrays = [item['array'] for item in batch]

    # Original fl_ce_color.py behavior - no contrast merging
    if contrast_type is None:
        return {
            'clean': target_images,
            'noisy': target_images  # Same as input for original behavior
        }

    # Create source (noisy) images based on contrast type
    if contrast_type == 'pair':
        if merge_type == 'uniform':
            # Random sampling at batch level for uniform merge
            source_arrays = []
            for arr in arrays:
                # Randomly choose transformation for each image
                case = random.randint(0, 3)
                if case == 0:
                    # Keep original
                    source_arr = arr.copy()
                elif case == 1:
                    # color_from -> color_to
                    source_arr = color_transform_standard(
                        arr, color_from, color_to,
                        use_mask_only=False,
                        bidirectional=False
                    )
                elif case == 2:
                    # color_to -> color_from
                    source_arr = color_transform_standard(
                        arr, color_to, color_from,
                        use_mask_only=False,
                        bidirectional=False
                    )
                else:
                    # Bidirectional swap
                    source_arr = color_transform_standard(
                        arr, color_from, color_to,
                        use_mask_only=False,
                        bidirectional=True
                    )
                source_arrays.append(source_arr)
        else:
            # Mask merge - convert to grayscale
            source_arrays = [
                color_transform_standard(
                    arr, color_from, color_to,
                    use_mask_only=True,
                    bidirectional=False
                )
                for arr in arrays
            ]
    elif contrast_type == 'all':
        # Full grayscale conversion
        source_arrays = [
            color_transform_standard(
                arr, None, None,
                no_color=True
            )
            for arr in arrays
        ]
    else:  # value
        # Standard colors without intensity
        source_arrays = [
            color_transform_standard(
                arr, None, None,
                no_value=True
            )
            for arr in arrays
        ]

    # Convert source arrays to tensors with same normalization as target
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    source_images = torch.stack([
        transform(Image.fromarray(arr))
        for arr in source_arrays
    ])

    return {
        'clean': target_images,  # Original (target)
        'noisy': source_images   # Contrast-merged (source)
    }

def eval_collate_fn(batch, contrast_type='pair', color_from=None, color_to=None,
                    merge_type='mask', seed=None):
    """
    Evaluation collate function with deterministic contrast merging.

    Same functionality as train_collate_fn but uses fixed random seed
    for reproducible evaluation.

    Args:
        batch: List of dictionaries from dataset
        contrast_type (str): Type of contrast to compute
        color_from (str): Source color for pair contrast
        color_to (str): Target color for pair contrast
        merge_type (str): How to merge colors
        seed (int, optional): Random seed for reproducible sampling

    Returns:
        dict: Contains:
            - clean: Original images [B,C,H,W] (target)
            - noisy: Contrast-merged images [B,C,H,W] (source)
    """
    # Create separate RNG for evaluation
    eval_rng = random.Random(seed) if seed is not None else random.Random()

    # Stack target images
    target_images = torch.stack([item['image'] for item in batch])
    arrays = [item['array'] for item in batch]

    # Original fl_ce_color.py behavior - no contrast merging
    if contrast_type is None:
        return {
            'clean': target_images,
            'noisy': target_images  # Same as input for original behavior
        }

    # Create source images with fixed random seed for uniform merge
    if contrast_type == 'pair':
        if merge_type == 'uniform':
            source_arrays = []
            for arr in arrays:
                case = eval_rng.randint(0, 3)  # Use seeded RNG
                if case == 0:
                    source_arr = arr.copy()
                elif case == 1:
                    source_arr = color_transform_standard(
                        arr, color_from, color_to,
                        use_mask_only=False,
                        bidirectional=False
                    )
                elif case == 2:
                    source_arr = color_transform_standard(
                        arr, color_to, color_from,
                        use_mask_only=False,
                        bidirectional=False
                    )
                else:
                    source_arr = color_transform_standard(
                        arr, color_from, color_to,
                        use_mask_only=False,
                        bidirectional=True
                    )
                source_arrays.append(source_arr)
        else:
            # Deterministic mask merge
            source_arrays = [
                color_transform_standard(
                    arr, color_from, color_to,
                    use_mask_only=True,
                    bidirectional=False
                )
                for arr in arrays
            ]
    elif contrast_type == 'all':
        source_arrays = [
            color_transform_standard(
                arr, None, None,
                no_color=True
            )
            for arr in arrays
        ]
    else:  # value
        source_arrays = [
            color_transform_standard(
                arr, None, None,
                no_value=True
            )
            for arr in arrays
        ]

    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    source_images = torch.stack([
        transform(Image.fromarray(arr))
        for arr in source_arrays
    ])

    return {
        'clean': target_images,  # Original (target)
        'noisy': source_images   # Contrast-merged (source)
    }

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
                 cond_entropy_bits_clean=0.0, contrast_type='pair',
                 color_from=None, color_to=None, merge_type='mask',
                 mtest_loader=None, pretrained_model=None):
    """
    Train MAE model and compute functional load (FL) between clean and noisy channels.

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
        model (nn.Module): MAE model
        optimizer (Optimizer): Model optimizer
        scheduler (LRScheduler): Learning rate scheduler
        train_loader (DataLoader): Training data loader
        dev_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        save_epoch_interval (int): Save checkpoints every N epochs
        result_dir (str): Directory to save results
        device (torch.device): Device to train on
        pixel_variance (float): Assumed pixel noise variance σ²
        num_save_images (int): Number of images to save
        save_image_interval (int): Save images every N epochs
        cond_entropy_bits_clean (float): Clean channel conditional entropy
        contrast_type (str): Type of contrast to analyze
        color_from (str): First color in pair contrast
        color_to (str): Second color in pair contrast
        merge_type (str): How to merge colors
        mtest_loader (DataLoader, optional): Test data loader
        pretrained_model (str, optional): Path to pretrained model
    """
    # Setup logging and directories
    overwrite_result_directory(result_dir, args.overwrite)
    logger = init_logger(os.path.join(args.result_dir, "report.log"))
    logger.info(args)
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    # Create stats directory and CSV files
    stats_dir = os.path.join(result_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    csv_files = {
        'train': os.path.join(stats_dir, 'train.csv'),
        'dev': os.path.join(stats_dir, 'dev.csv'),
        'test': os.path.join(stats_dir, 'test.csv')
    }

    # Initialize CSV files with headers
    csv_headers = ['epoch', 'dataset', 'kl_divergence', 'mse_per_pixel',
                  'cond_entropy_bits', 'cond_entropy_bits_diff',
                  'contrast_type', 'color_from', 'color_to', 'merge_type']

    for csv_file in csv_files.values():
        with open(csv_file, 'w', newline='') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            csv_writer.writerow(csv_headers)

    # Log configuration
    if contrast_type is None:
        logger.info('Training MAE model for image reconstruction (clean channel without contrast merge)')
    else:
        logger.info('Training MAE model for contrast analysis')
        logger.info("\nContrast Configuration:")
        logger.info(f"Contrast type: {contrast_type}")
        if contrast_type == 'pair':
            logger.info(f"Colors: {color_from} -> {color_to}")
            logger.info(f"Merge type: {merge_type}")
    logger.info(f"Clean channel entropy: {cond_entropy_bits_clean:.6f} bits/pixel")

    # Load pretrained model if specified
    if pretrained_model:
        checkpoint = torch.load(pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded pretrained model from {pretrained_model}")

    best_dev_loss = float('inf')
    best_dev_epoch = 0
    epoch = 0

    # Evaluate pretrained model if starting from epoch 0
    if epoch == 0 and pretrained_model:
        epoch += 1
        model.eval()
        info_table = []

        with torch.no_grad():
            # Evaluate on all datasets
            for loader, name, csv_file in [
                (train_loader, 'train_set', csv_files['train']),
                (dev_loader, 'dev_set', csv_files['dev']),
                (mtest_loader, 'test_set', csv_files['test'])
            ]:
                if loader is None:
                    continue

                # Initialize statistics
                eval_loss = 0.0
                total_mse = 0.0
                total_pixels = 0

                for batch in tqdm.tqdm(loader, ascii=True, ncols=50):
                    target = batch['clean'].to(device)  # Original images
                    source = batch['noisy'].to(device)  # Contrast-merged images

                    # Get reconstructions
                    outputs = model(source)
                    reconstructed = unpatchify(outputs.logits, patch_size=16)

                    # Calculate MSE
                    mse_loss = F.mse_loss(reconstructed, target, reduction='sum')
                    num_pixels = target.numel()

                    # Calculate KL divergence: (X-Y)²/(2σ²)
                    kl_loss = mse_loss/(2*pixel_variance)
                    kl_per_pixel = kl_loss/num_pixels

                    # Calculate differential conditional entropy
                    # Negative conditional entropy indicates:
                    # - Reconstruction precision better than reference scale
                    # - Model achieves sub-unit accuracy in prediction
                    # - Common in continuous domains with good reconstruction
                    # - Conditional entropy is relative uncertainty
                    # Calculate conditional entropy directly: -log P(X|Y) = (X-Y)²/(2σ²) + log(2πσ²)/2
                    cond_entropy = mse_loss/(2*pixel_variance) + \
                                    num_pixels*np.log(2*np.pi*pixel_variance)/2

                    # Convert to bits and per-pixel rate
                    bits_per_pixel = cond_entropy / (num_pixels * np.log(2))

                    eval_loss += kl_per_pixel.item()
                    total_mse += mse_loss.item()
                    total_pixels += num_pixels

                # Calculate average metrics
                eval_loss /= len(loader)
                mse_per_pixel = total_mse / total_pixels

                # Calculate conditional entropy rate in bits/pixel
                cond_entropy = total_mse/(2*pixel_variance) + \
                                total_pixels*np.log(2*np.pi*pixel_variance)/2
                cond_entropy_bits = cond_entropy / (total_pixels * np.log(2))

                # Calculate entropy difference from clean channel
                cond_entropy_bits_diff = cond_entropy_bits - cond_entropy_bits_clean

                # Log to info table
                info_table.append([
                    0,  # epoch 0
                    name,
                    f"{eval_loss:.6e}",
                    f"{mse_per_pixel:.6e}",
                    f"{cond_entropy_bits:.6e}",
                    f"{cond_entropy_bits_diff:+.6e}"  # Show diff with sign
                ])

                # Log to CSV
                with open(csv_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f, lineterminator='\n')
                    csv_writer.writerow([
                        0, name, f"{eval_loss:.6e}", f"{mse_per_pixel:.6e}",
                        f"{cond_entropy_bits:.6e}", f"{cond_entropy_bits_diff:+.6e}",
                        contrast_type, color_from, color_to, merge_type
                    ])

                # Log to tensorboard
                phase = name.replace('_set', '')
                writer.add_scalar(f'KLDivergence/{phase}', eval_loss, 0)
                writer.add_scalar(f'MSE/{phase}', mse_per_pixel, 0)
                writer.add_scalar(f'ConditionalEntropyRate_bits/{phase}', cond_entropy_bits, 0)
                writer.add_scalar(f'ConditionalEntropyRate_bits_diff/{phase}', cond_entropy_bits_diff, 0)

                # In epoch 0 evaluation block, after metric calculations:
                # Save images for all datasets at epoch 0
                if loader in [train_loader, dev_loader, mtest_loader]:
                    batch = next(iter(loader))
                    target = batch['clean'][:num_save_images]
                    source = batch['noisy'][:num_save_images]
                    reconstructed = model(source.to(device)).logits[:num_save_images]
                    reconstructed = unpatchify(reconstructed, patch_size=16)

                    # Determine prefix based on loader
                    prefix = {
                        train_loader: 'Train',
                        dev_loader: 'Val',
                        mtest_loader: 'Test'
                    }[loader]

                    writer.add_images(f'{prefix}/Original', denormalize(target), 0)
                    writer.add_images(f'{prefix}/Reconstructed', denormalize(reconstructed), 0)
                    writer.add_images(f'{prefix}/Source', denormalize(source), 0)
            # Print reference entropy and metrics table
            logger.info(f"\nClean channel entropy: {cond_entropy_bits_clean:.6f} bits/pixel")
            logger.info("\n" + tabulate.tabulate(
                info_table,
                headers=['epoch', 'dataset', 'kl_divergence', 'mse_per_pixel',
                        'cond_entropy_bits', 'cond_entropy_bits_diff'],
                floatfmt='.6e',
                tablefmt='rst'
            ))
            # Exit if only evaluating pretrained model
            if num_epochs == 0:
                writer.close()
                return

    while epoch < num_epochs:
        # Training phase
        model.train()
        train_kl_loss = 0.0
        total_mse = 0.0
        total_pixels = 0
        info_table = []

        for batch in tqdm.tqdm(train_loader, ascii=True, ncols=50):
            target = batch['clean'].to(device)
            source = batch['noisy'].to(device)

            # Forward pass
            outputs = model(source)
            reconstructed = unpatchify(outputs.logits, patch_size=16)

            # Calculate statistics exactly as in evaluation
            # MSE loss: Σ(X-Y)²
            mse_loss = F.mse_loss(reconstructed, target, reduction='sum')
            num_pixels = target.numel()

            # KL divergence: (X-Y)²/(2σ²)
            kl_loss = mse_loss/(2*pixel_variance)
            kl_per_pixel = kl_loss/num_pixels

            # Conditional entropy: (X-Y)²/(2σ²) + log(2πσ²)/2
            cond_entropy = mse_loss/(2*pixel_variance) + \
                          num_pixels*np.log(2*np.pi*pixel_variance)/2
            bits_per_pixel = cond_entropy / (num_pixels * np.log(2))

            # Optimization step using KL divergence
            optimizer.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
            optimizer.step()

            # Accumulate statistics
            train_kl_loss += kl_per_pixel.item()
            total_mse += mse_loss.item()
            total_pixels += num_pixels

            if torch.isnan(kl_loss):
                raise ValueError("NaN detected in loss calculation")

        # Calculate training metrics
        train_kl_loss /= len(train_loader)  # Average KL divergence per pixel
        train_mse_per_pixel = total_mse / total_pixels

        # Calculate conditional entropy rate in bits/pixel
        train_cond_entropy = total_mse/(2*pixel_variance) + \
                            total_pixels*np.log(2*np.pi*pixel_variance)/2
        train_cond_entropy_bits = train_cond_entropy / (total_pixels * np.log(2))

        # Calculate difference from clean channel entropy
        train_entropy_bits_diff = train_cond_entropy_bits - cond_entropy_bits_clean

        # Log training metrics
        info_table.append([
            epoch,
            "train_set",
            f"{train_kl_loss:.6e}",
            f"{train_mse_per_pixel:.6e}",
            f"{train_cond_entropy_bits:.6e}",
            f"{train_entropy_bits_diff:+.6e}"  # Show diff with sign
        ])

        # Log to CSV with Unix line endings
        with open(csv_files['train'], 'a', newline='') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            csv_writer.writerow([
                epoch, 'train_set',
                f"{train_kl_loss:.6e}", f"{train_mse_per_pixel:.6e}",
                f"{train_cond_entropy_bits:.6e}", f"{train_entropy_bits_diff:+.6e}",
                contrast_type, color_from, color_to, merge_type
            ])

        # Log training metrics to tensorboard
        writer.add_scalar('KLDivergence/train', train_kl_loss, epoch)
        writer.add_scalar('MSE/train', train_mse_per_pixel, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits/train', train_cond_entropy_bits, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits_diff/train', train_entropy_bits_diff, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Evaluation phase
        model.eval()
        eval_kl_loss = 0.0
        total_mse = 0.0
        total_pixels = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(dev_loader, ascii=True, ncols=50):
                target = batch['clean'].to(device)
                source = batch['noisy'].to(device)

                # Forward pass
                outputs = model(source)
                reconstructed = unpatchify(outputs.logits, patch_size=16)

                # Calculate same statistics as training
                mse_loss = F.mse_loss(reconstructed, target, reduction='sum')
                num_pixels = target.numel()

                kl_loss = mse_loss/(2*pixel_variance)
                kl_per_pixel = kl_loss/num_pixels

                cond_entropy = mse_loss/(2*pixel_variance) + \
                              num_pixels*np.log(2*np.pi*pixel_variance)/2
                bits_per_pixel = cond_entropy / (num_pixels * np.log(2))

                eval_kl_loss += kl_per_pixel.item()
                total_mse += mse_loss.item()
                total_pixels += num_pixels

        # Calculate validation metrics
        eval_kl_loss /= len(dev_loader)
        eval_mse_per_pixel = total_mse / total_pixels

        eval_cond_entropy = total_mse/(2*pixel_variance) + \
                           total_pixels*np.log(2*np.pi*pixel_variance)/2
        eval_cond_entropy_bits = eval_cond_entropy / (total_pixels * np.log(2))
        eval_entropy_bits_diff = eval_cond_entropy_bits - cond_entropy_bits_clean

        # Log validation metrics
        info_table.append([
            epoch,
            "dev_set",
            f"{eval_kl_loss:.6e}",
            f"{eval_mse_per_pixel:.6e}",
            f"{eval_cond_entropy_bits:.6e}",
            f"{eval_entropy_bits_diff:+.6e}"
        ])

        # Log validation to CSV
        with open(csv_files['dev'], 'a', newline='') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            csv_writer.writerow([
                epoch, 'dev_set',
                f"{eval_kl_loss:.6e}", f"{eval_mse_per_pixel:.6e}",
                f"{eval_cond_entropy_bits:.6e}", f"{eval_entropy_bits_diff:+.6e}",
                contrast_type, color_from, color_to, merge_type
            ])

        # Log validation metrics to tensorboard
        writer.add_scalar('KLDivergence/dev', eval_kl_loss, epoch)
        writer.add_scalar('MSE/dev', eval_mse_per_pixel, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits/dev', eval_cond_entropy_bits, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits_diff/dev', eval_entropy_bits_diff, epoch)

        # Test monitoring with same calculations
        if mtest_loader:
            model.eval()
            test_kl_loss = 0.0
            total_mse = 0.0
            total_pixels = 0

            with torch.no_grad():
                for batch in tqdm.tqdm(mtest_loader, ascii=True, ncols=50):
                    target = batch['clean'].to(device)
                    source = batch['noisy'].to(device)

                    outputs = model(source)
                    reconstructed = unpatchify(outputs.logits, patch_size=16)

                    # Calculate same statistics
                    mse_loss = F.mse_loss(reconstructed, target, reduction='sum')
                    num_pixels = target.numel()

                    kl_loss = mse_loss/(2*pixel_variance)
                    kl_per_pixel = kl_loss/num_pixels

                    cond_entropy = mse_loss/(2*pixel_variance) + \
                                  num_pixels*np.log(2*np.pi*pixel_variance)/2
                    bits_per_pixel = cond_entropy / (num_pixels * np.log(2))

                    test_kl_loss += kl_per_pixel.item()
                    total_mse += mse_loss.item()
                    total_pixels += num_pixels

            # Calculate test metrics
            test_kl_loss /= len(mtest_loader)
            test_mse_per_pixel = total_mse / total_pixels

            test_cond_entropy = total_mse/(2*pixel_variance) + \
                               total_pixels*np.log(2*np.pi*pixel_variance)/2
            test_cond_entropy_bits = test_cond_entropy / (total_pixels * np.log(2))
            test_entropy_bits_diff = test_cond_entropy_bits - cond_entropy_bits_clean

            # Log test metrics
            info_table.append([
                epoch,
                "test_set",
                f"{test_kl_loss:.6e}",
                f"{test_mse_per_pixel:.6e}",
                f"{test_cond_entropy_bits:.6e}",
                f"{test_entropy_bits_diff:+.6e}"
            ])

            # Log test to CSV
            with open(csv_files['test'], 'a', newline='') as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                csv_writer.writerow([
                    epoch, 'test_set',
                    f"{test_kl_loss:.6e}", f"{test_mse_per_pixel:.6e}",
                    f"{test_cond_entropy_bits:.6e}", f"{test_entropy_bits_diff:+.6e}",
                    contrast_type, color_from, color_to, merge_type
                ])

            # Log test metrics to tensorboard
            writer.add_scalar('KLDivergence/test', test_kl_loss, epoch)
            writer.add_scalar('MSE/test', test_mse_per_pixel, epoch)
            writer.add_scalar('ConditionalEntropyRate_bits/test', test_cond_entropy_bits, epoch)
            writer.add_scalar('ConditionalEntropyRate_bits_diff/test', test_entropy_bits_diff, epoch)

        # Save images in target/reconstructed/source order
        if epoch % save_image_interval == 0:
            for prefix, loader in [('Train', train_loader),
                                ('Val', dev_loader),
                                ('Test', mtest_loader)]:
                if loader is None:  # Skip if loader doesn't exist
                    continue

                batch = next(iter(loader))
                target = batch['clean'][:num_save_images]
                source = batch['noisy'][:num_save_images]
                reconstructed = model(source.to(device)).logits[:num_save_images]
                reconstructed = unpatchify(reconstructed, patch_size=16)

                writer.add_images(f'{prefix}/Original', denormalize(target), epoch)
                writer.add_images(f'{prefix}/Reconstructed', denormalize(reconstructed), epoch)
                writer.add_images(f'{prefix}/Source', denormalize(source), epoch)

        # Print reference entropy and metrics table
        logger.info(f"\nClean channel entropy: {cond_entropy_bits_clean:.6f} bits/pixel")
        logger.info("\n" + tabulate.tabulate(
            info_table,
            headers=['epoch', 'dataset', 'kl_divergence', 'mse_per_pixel',
                    'cond_entropy_bits', 'cond_entropy_bits_diff'],
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

        # Save best model based on validation KL divergence
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

# Contrast arguments
parser.add_argument('--contrast_type', type=str,
                   choices=['pair', 'all', 'value', None], default=None,
                   help='Type of contrast to compute FL for: pair (two colors), '
                        'all (full grayscale), or value (no intensity);'
                        'None for original fl_ce_color.py behavior')
parser.add_argument('--color_from', type=str, default=None,
                   help='First color in pair for pair contrast type')
parser.add_argument('--color_to', type=str, default=None,
                   help='Second color in pair for pair contrast type')
parser.add_argument('--merge_type', type=str,
                   choices=['uniform', 'mask'], default='mask',
                   help='How to merge colors: uniform sampling or grayscale mask')

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
parser.add_argument('--pretrained_model', type=str, default=None,
                   help='Path to pretrained model checkpoint')

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

# Statistical arguments
parser.add_argument('--pixel_variance', type=float, default=0.05,
                   help='Assumed variance for pixel distributions (default: 0.05 close to ImageNet variance)')
parser.add_argument('--cond_entropy_bits_clean', type=float, default=0.0,
                   help='Clean channel conditional entropy in bits/pixel')
parser.add_argument('--num_save_images', type=int, default=8,
                   help='Number of images to save for visualization')

# Other arguments
parser.add_argument('--seed', type=int, default=2025,
                   help='Random seed for reproducibility')
parser.add_argument('--gpu', type=str, default='auto',
                   help='GPU selection ("auto" or specific id)')
parser.add_argument('--result_dir', type=str, default='exp/fl_ce_imagenet/test/sample_imagenet',
                   help='Directory to save results')
parser.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing result directory')
parser.add_argument('--exit', action='store_true',
                   help='Exit after training without prompting to continue')

args = parser.parse_args()

set_seed(args.seed)
device = set_device(args.gpu)

# Create datasets and dataloaders
train_dataset = ImageNetContrastDataset(args.train_dir)
dev_dataset = ImageNetContrastDataset(args.dev_dir)

# Create train dataloader with contrast merging
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda batch: train_collate_fn(
        batch,
        contrast_type=args.contrast_type,
        color_from=args.color_from,
        color_to=args.color_to,
        merge_type=args.merge_type
    )
)

# Create validation dataloader with fixed seed for reproducibility
dev_loader = DataLoader(
    dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda batch: eval_collate_fn(
        batch,
        contrast_type=args.contrast_type,
        color_from=args.color_from,
        color_to=args.color_to,
        merge_type=args.merge_type,
        seed=args.seed
    )
)

# Create test dataset and loader if specified
mtest_loader = None
if args.mtest_dir:
    mtest_dataset = ImageNetContrastDataset(args.mtest_dir)
    mtest_loader = DataLoader(
        mtest_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: eval_collate_fn(
            batch,
            contrast_type=args.contrast_type,
            color_from=args.color_from,
            color_to=args.color_to,
            merge_type=args.merge_type,
            seed=args.seed
        )
    )

# Create model
model = create_mae_model(
    image_size=args.image_size,
    patch_size=args.patch_size,
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    channels=3,  # Always 3 for RGB images
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

# Train model
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
    cond_entropy_bits_clean=args.cond_entropy_bits_clean,
    contrast_type=args.contrast_type,
    color_from=args.color_from,
    color_to=args.color_to,
    merge_type=args.merge_type,
    num_save_images=args.num_save_images,
    save_image_interval=args.save_image_interval,
    mtest_loader=mtest_loader,
    pretrained_model=args.pretrained_model
)
