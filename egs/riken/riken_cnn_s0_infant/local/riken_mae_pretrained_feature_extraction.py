"""
Extract features from pretrained MAE (Masked Autoencoder) model.

This script extracts the class token embeddings from a pretrained MAE model for each input image
and saves them as an NPY file. It can also optionally save the reconstructed images.

It supports loading pretrained models from various sources:
- Facebook's pretrained MAE model ('base')
- Custom pretrained MAE models
- Training from scratch if no pretrained model is specified

The extracted features can be used for downstream tasks like classification or clustering.

Usage:
    python riken_mae_pretrained_feature_extraction.py
        --input_file path/to/input.npy
        --output_dir path/to/output_dir
        --pretrained_path path/to/pretrained_model.pt
        --output_file path/to/explicit_output.npy (optional)
        --save_reconstructions (to save reconstructed images)

Author: Based on work by bin-wu
Date: March 19, 2025
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import ViTMAEModel, ViTMAEConfig, ViTMAEForPreTraining
import GPUtil

# Reuse functions from the original script
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

# Reuse SpectrogramDataset from the original script
from torch.utils.data import Dataset
from torchvision import transforms

class SpectrogramDataset(Dataset):
    """
    Dataset for loading and preprocessing spectrogram data for MAE feature extraction.

    Args:
        input_path (str): Path to input .npy file containing spectrograms
        transform (callable, optional): Additional transforms to apply

    Attributes:
        inputs (np.ndarray): Loaded spectrogram data
        transform (callable): Transform pipeline
        normalize (transforms.Normalize): ImageNet normalization
    """
    def __init__(self, input_path, transform=None):
        self.inputs = np.load(input_path)
        self.transform = transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def process_input(self, x):
        """Process single-channel spectrogram input"""
        # Convert to float32 if needed
        x = x.astype(np.float32)

        # Normalize to [0, 1]
        x = (x - x.min()) / (x.max() - x.min())

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

def create_mae_feature_extractor(args, logger, for_reconstruction=False):
    """
    Create MAE model for feature extraction or reconstruction.

    This function initializes a MAE model using the specified configuration, loading
    weights from a pretrained source if provided.

    Args:
        args: Command line arguments containing:
            pretrained_path (str): Path to pretrained model, 'base' for facebook/vit-mae-base, or None
            image_size (int): Size of input images (height and width)
            patch_size (int): Size of patches for the ViT model
            seq (bool): Whether to use 1D sequential patching
            dim (int): Hidden dimension size
            depth (int): Number of transformer layers
            heads (int): Number of attention heads
            mlp_dim (int): Dimension of MLP layer
            dropout (float): Dropout rate
            mask_ratio (float): Ratio of patches to mask during reconstruction
        logger: Logger instance for recording progress and errors
        for_reconstruction (bool): If True, return a full MAE model with decoder
                                  for image reconstruction; if False, return just
                                  the encoder for feature extraction

    Returns:
        nn.Module: Either a ViTMAEModel (encoder only) or ViTMAEForPreTraining (full model)
                  depending on the for_reconstruction parameter
    """
    model_class = ViTMAEForPreTraining if for_reconstruction else ViTMAEModel

    if args.pretrained_path == 'base':
        # Load pretrained model from Facebook
        logger.info(f"Loading pretrained model from 'facebook/vit-mae-base' as {model_class.__name__}")
        model = model_class.from_pretrained('facebook/vit-mae-base')
    elif args.pretrained_path:
        # Load from checkpoint saved by riken_mae_pretrained.py
        logger.info(f"Loading pretrained model from '{args.pretrained_path}' as {model_class.__name__}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')

        # Check if the checkpoint contains a ViTMAEForPreTraining or just the state dict
        if 'model' in checkpoint:
            # Create a new MAE model first
            config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
            model = model_class.from_pretrained('facebook/vit-mae-base', config=config)

            if for_reconstruction:
                # For reconstruction, load the entire model
                model.load_state_dict(checkpoint['model'], strict=False)
                logger.info("Loaded full MAE model from checkpoint")
            else:
                # For feature extraction, we only need the encoder part
                encoder_state_dict = {}
                for key, value in checkpoint['model'].items():
                    # Only keep keys that start with 'vit.' (the encoder part)
                    if key.startswith('vit.'):
                        # Remove 'vit.' prefix to match ViTMAEModel keys
                        encoder_state_dict[key[4:]] = value

                # Load the extracted encoder state dict
                model.load_state_dict(encoder_state_dict, strict=False)
                logger.info("Loaded encoder weights from MAE pretrained checkpoint")
        else:
            # In case it's already a model checkpoint
            model = model_class.from_pretrained('facebook/vit-mae-base')
            model.load_state_dict(checkpoint)
            logger.info("Loaded weights directly from checkpoint")
    else:
        # Create model from scratch
        logger.info(f"Creating {model_class.__name__} from scratch (no pretrained weights)")
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
            mask_ratio=args.mask_ratio if for_reconstruction else 0.0  # No masking for feature extraction
        )
        model = model_class(config)

    return model

def unpatchify(x, patch_size=16):
    """
    Convert patch representations back to original image format.

    This function takes a batch of patch-based representations (output from the MAE decoder)
    and reconstructs them into full images.

    Args:
        x (torch.Tensor): Tensor of shape [B, N, P*P*C] where:
            B = batch size
            N = number of patches (typically H/p * W/p)
            P = patch size
            C = number of channels (typically 3)
        patch_size (int): Size of each patch (P in the formula above)

    Returns:
        torch.Tensor: Reconstructed images of shape [B, C, H, W]
                     where H = W = √N * patch_size
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

def extract_features(args, logger):
    """
    Extract features from pretrained MAE model and optionally save reconstructed images.

    This function processes the input spectrograms through a MAE model, extracts the
    class token embeddings, and saves them to an NPY file. If requested, it also
    generates and saves reconstructed images.

    Args:
        args: Command line arguments containing:
            input_file (str): Path to input NPY file containing spectrograms
            output_dir (str): Directory to save output files
            output_file (str): Explicit path for output file (optional)
            save_reconstructions (bool): Whether to generate and save reconstructions
            batch_size (int): Batch size for processing
            num_workers (int): Number of data loading workers
            gpu (str): GPU selection (number or 'auto')
            pretrained_path (str): Path to pretrained model
            image_size (int): Size of input images
            patch_size (int): Size of patches
            mask_ratio (float): Ratio of patches to mask during reconstruction
        logger: Logger instance for recording progress and errors

    Returns:
        None: Results are saved to files specified by output_file or generated from
              model_name and input_file name
    """
    device = set_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Create model for feature extraction
    encoder_model = create_mae_feature_extractor(args, logger, for_reconstruction=False).to(device)
    encoder_model.eval()
    logger.info("Feature extraction model loaded")

    # Create model for reconstruction if needed
    if args.save_reconstructions:
        recon_model = create_mae_feature_extractor(args, logger, for_reconstruction=True).to(device)
        recon_model.eval()
        logger.info("Reconstruction model loaded")

    # Get the input file name without path and extension for output naming
    if args.input_file:
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
    else:
        input_name = "unknown_input"

    # Get the model name for output naming
    if args.pretrained_path == 'base':
        model_name = "vit-mae-base"
    elif args.pretrained_path:
        model_name = os.path.splitext(os.path.basename(args.pretrained_path))[0]
    else:
        model_name = "untrained"

    # Determine output file paths
    if args.output_file:
        features_output_file = args.output_file
        recon_output_file = os.path.splitext(args.output_file)[0] + "_recon.npy"
    else:
        features_output_file = os.path.join(args.output_dir, f"mae_feat_{model_name}_{input_name}.npy")
        recon_output_file = os.path.join(args.output_dir, f"mae_feat_{model_name}_{input_name}_recon.npy")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(features_output_file), exist_ok=True)

    # Create dataset and dataloader
    transform = transforms.Resize((args.image_size, args.image_size))
    dataset = SpectrogramDataset(args.input_file, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Features output file: {features_output_file}")
    if args.save_reconstructions:
        logger.info(f"Reconstructions output file: {recon_output_file}")

    # Extract features
    features = []
    reconstructions = [] if args.save_reconstructions else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch.to(device)

            # Get encoder outputs for feature extraction
            outputs = encoder_model(images, output_hidden_states=True)

            # Get the class token embeddings ([CLS] token)
            # The class token is the first token (index 0) of the last hidden state
            cls_token = outputs.last_hidden_state[:, 0, :]

            # Move to CPU and convert to numpy
            features.append(cls_token.cpu().numpy())

            # Generate reconstructions if requested
            if args.save_reconstructions:
                # Get reconstructed images
                recon_outputs = recon_model(images)

                # Convert reconstruction logits to images
                # The logits are of shape [B, N, P*P*C] where N is the number of patches
                recon_images = unpatchify(recon_outputs.logits, patch_size=args.patch_size)

                # Normalize to [0, 1] range for better visualization
                recon_images = (recon_images - recon_images.min()) / (recon_images.max() - recon_images.min())

                # Move to CPU and convert to numpy
                reconstructions.append(recon_images.cpu().numpy())

    # Concatenate all features and save to file
    features = np.concatenate(features, axis=0)
    logger.info(f"Extracted features shape: {features.shape}")
    np.save(features_output_file, features)
    logger.info(f"Saved features to {features_output_file}")

    # Save reconstructions if requested
    if args.save_reconstructions:
        reconstructions = np.concatenate(reconstructions, axis=0)
        logger.info(f"Reconstructed images shape: {reconstructions.shape}")
        np.save(recon_output_file, reconstructions)
        logger.info(f"Saved reconstructions to {recon_output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract features from pretrained MAE model')

    # Input/Output arguments
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument('--input_file', type=str,
                        default='exp/data/division_nas5_b2_mae_pretrained_all_days/dev_input.npy',
                        help='Path to the input file containing spectrograms')
    io_group.add_argument('--output_dir', type=str, default='exp/tests/mae_feat',
                        help='Directory to save extracted features')
    io_group.add_argument('--output_file', type=str, default=None,
                        help='Explicit path for output file. If not specified, it will be generated automatically')
    io_group.add_argument('--save_reconstructions', action='store_true',
                        help='Save reconstructed images alongside features')

    # Model arguments (keep original parameters)
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
    model_group.add_argument('--mask_ratio', type=float, default=0,
                            help='Ratio of patches to mask during reconstruction')
    model_group.add_argument('--pretrained_path', type=str, default=None,
                            help="Path to pretrained model, 'base' for facebook/vit-mae-base, or None")

    # Processing arguments
    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument('--batch_size', type=int, default=64,
                           help='Batch size for feature extraction')
    proc_group.add_argument('--num_workers', type=int, default=4,
                           help='Number of workers for data loading')
    proc_group.add_argument('--gpu', type=str, default='auto',
                           help="GPU selection: number for specific GPU or 'auto' for least used")
    proc_group.add_argument('--seed', type=int, default=2020,
                           help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    logger = init_logger(os.path.join(args.output_dir, "feature_extraction.log"))
    logger.info(f"Arguments: {args}")

    # Extract features
    extract_features(args, logger)

if __name__ == '__main__':
    main()