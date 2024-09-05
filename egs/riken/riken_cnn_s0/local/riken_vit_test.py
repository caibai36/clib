# Implemented by bin-wu at 17:29 on 18 April 2024
# Updated to use Vision Transformer (ViT) at 21:31 on 12 August 2024

import os
import sys
import datetime
import logging
import glob
import shutil

import math
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

# Reference: https://github.com/lucidrains/vit-pytorch
#
# Updated by bin-wu on 2024/08/12 at 18:54:
# - Added comments and docstrings
# - Implemented returning attention weights
# - Enhanced the forward method to handle labels, among other improvements

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    """
    Convert a single value to a pair.
    If t is already a pair, return it as is.
    """
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    """
    Implements a feed-forward network with GELU activation and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Implements multi-head self-attention mechanism.

    Args:
        dim (int): The input dimension
            Note: The dim shared between the transformer blocks and between attention and feedforward sub-blocks
            The d_model (in transformer terminology) would be inner_dim = dim_head * heads
            The data flow within the transformer follows the pattern: `dim => inner_dim => ... => inner_dim => dim`.
            When `dim` equals `inner_dim`, the implementation behaves as a standard transformer.
        heads (int): Number of attention heads (default 8)
        dim_head (int): Dimension of each attention head (default 64)
        dropout (float): Dropout probability
        return_attention (bool): Whether to return attention weights (default False)

    Shape:
        - Input: (batch_size, seq_len, dim)
        - Output: (batch_size, seq_len, dim), attention weights (optional)
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., return_attention = False):
        super().__init__()
        inner_dim = dim_head * heads # d_model of transformer
        project_out = not (heads == 1 and dim_head == dim)

        self.return_attention = return_attention
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # Single linear layer to compute query, key, and value
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim)
            torch.Tensor: Attention weights if return_attention is True, else None
        """
        x = self.norm(x) # pre_norm with shape: b n input_dim

        # Project input into query, key, and value representations with different transformation
        # Rearrange q, k, v to separate the heads
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 3 b n (h d)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # d_model=hxd=headxhead_dim; n=hxw=num_patches_heightxnum_patches_width

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b h n d x b h d n -> b h n n

        # Apply softmax and masking to get attention weights
        attn = self.attend(dots)
        attention_weights = attn if self.return_attention else None
        attn = self.dropout(attn)

        # Apply attention weights to values and concatenate heads
        out = torch.matmul(attn, v) # b h n n x b h n d -> b h n d
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attention_weights

class Transformer(nn.Module):
    """
    Implements a transformer block with multiple layers of attention and feed-forward networks.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., return_attention = False):
        super().__init__()
        self.return_attention = return_attention
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, return_attention = return_attention),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        attention_weights = []
        for attn, ff in self.layers:
            attn_out, weights = attn(x)
            x = attn_out + x  # Attention with residual connection
            x = ff(x) + x    # Feed-forward with residual connection
            if self.return_attention:
                attention_weights.append(weights)

        return self.norm(x), attention_weights

class ViT(nn.Module):
    """
    Implementation of the Vision Transformer (ViT) model.

    Args:
        image_size (int, tuple): Image size. If you have rectangular images, make sure your image size is the maximum of the width and height.
        patch_size (int, tuple): Size of patches. `image_size` must be divisible by `patch_size`.
                          The number of patches is: `n = (image_size // patch_size) ** 2` and `n` must be greater than 16.
        num_classes (int): Number of classes to classify.
        dim (int): The dimensionality that is shared across the transformer blocks and shared between attention and linear sub-blocks.
            Within each attention sub-block, the dimension dim is first transformed into dim_head * heads for processing,
            and then converted back to dim after the attention operations are completed.
            If dim equals dim_head * heads, the implementation behaves as a standard transformer.
        depth (int): Number of Transformer blocks.
        heads (int): Number of heads in Multi-head Attention layer. (d_model = dim_head * heads)
        mlp_dim (int): Dimension of the MLP (FeedForward) layer.
        pool (str): Either 'cls' token pooling or 'mean' pooling, used for projection head processing the transformer output.
        channels (int, optional): Number of image's channels. Default is 3.
        dim_head (int, optional): Dimension of each attention head. Default is 64. (d_model = dim_head * heads)
        dropout (float, optional): Dropout rate. Default is 0.
        emb_dropout (float, optional): Embedding dropout rate. Default is 0.
        return_attention (bool, optional): Whether to return attention weights. Default is False.
            When `return_attention` is True, the forward pass returns a list of attention weight tensors,
            with a length equal to `num_blocks`, for visualization;
            each tensor in the list has the shape [batch_size, num_heads, seq_len, seq_len].
            If `return_attention` is False, the forward function returns None for attention weights.
        return_logits (bool, optional): Whether to return logits instead of probabilities. Default is False.
        return_latent (bool, optional): Whether to return latent logits (with dimension of dim) instead of probabilities. Default is False.

    Attributes:
        to_patch_embedding (nn.Sequential): Converts image to patch embeddings.
        pos_embedding (nn.Parameter): Positional embedding for patches.
        cls_token (nn.Parameter): Learnable classification token.
        dropout (nn.Dropout): Dropout layer.
        transformer (Transformer): Transformer encoder.
        pool (str): Pooling type ('cls' or 'mean').
        to_latent (nn.Identity): Identity layer for latent representation.
        mlp_head (nn.Linear): Final classification head.
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., return_attention = False, return_logits = False, return_latent=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.return_attention = return_attention
        self.return_logits = return_logits
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # Flatten
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, return_attention)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.return_latent = return_latent

    def forward(self, img, y=None):
        """
        Forward pass of the Vision Transformer.

        Args:
            img (torch.Tensor): Input image tensor with one of the following shapes:
                - (batch_size, channels, height, width): Standard image input.
                - (batch_size, height, width): Grayscale image with a single channel.
                - (batch_size, 257, 256): Spectrogram image with an additional dimension for the DC component.
                    One dimension will be removed during processing.
            y (torch.Tensor, optional): Ground truth labels for output, of shape (batch_size,). Defaults to None.

        Returns:
            If y is provided:
                tuple: A tuple containing:
                    - logits or probs (torch.Tensor): Logits if return_logits is True, else probabilities. Shape is (batch_size, num_classes).
                    - loss (torch.Tensor): The computed loss.
                    - accuracy (list): A list of accuracies for each sample in the batch.
                    - attention_weights (list or None): A list of attention weight tensors if return_attention is True, else None.
            If y is not provided:
                tuple: A tuple containing:
                    - logits or probs (torch.Tensor): Logits if return_logits is True, else probabilities. Shape is (batch_size, num_classes).
                    - attention_weights (list or None): A list of attention weight tensors if return_attention is True, else None.

            if return_latent is True, return latent logits (with dimension of dim) instead of probabilities. Shape is (batch_size, dim).

        Note:
            The attention_weights list has a length equal to `num_blocks`, with each element having shape
-           [batch_size, num_heads, seq_len, seq_len].
        """
        # Spectrogram image with an additional first dimension representing the DC component
        if img.shape[1:] == (257, 256): # Shape of image: (batch_size, 257, 256)
            img = img[:,1:]

        # Add dummy channel dimension if input is (batch_size, height, width)
        if len(img.shape) == 3:
            img = img.unsqueeze(1)

        x = self.to_patch_embedding(img) # Flatten 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] # Learnable
        x = self.dropout(x)

        x, attention_weights = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        latent = self.to_latent(x)
        logits = self.mlp_head(latent)

        # Determine whether to return logits or probabilities
        output = logits if self.return_logits else F.softmax(logits, dim=1)

        attention_weights = attention_weights if self.return_attention else None

        if self.return_latent:
            output = latent # Shape is (batch_size, dim)

        if y is not None:
            y = y.type(torch.LongTensor).to(y.device)
            loss = F.cross_entropy(logits, y)
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == y).float().tolist()
            return output, loss, accuracy, attention_weights
        else:
            return output, attention_weights

class OneStreamDataset(Dataset):
    """
    A PyTorch Dataset class for the one-stream data.

    Args:
        xs (numpy.ndarray): Input data, of shape (data_length, 257, 256).
        labels (numpy.ndarray): Labels corresponding to xs, of shape (data_length).
        train (bool): Whether the dataset is used for training or evaluation/development.
        apply_random_shift (bool): Whether to apply data argumentation of random shifts. Default is True.
                                   Each sample in the batch applies a new random shift within 5 pixels.
    """

    def __init__(self, xs, labels, train=True, apply_random_shift=True):
        self.xs = xs
        self.labels = labels
        self.train = train
        self.apply_random_shift = apply_random_shift

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        label = self.labels[idx]

        if self.train and self.apply_random_shift:
            ver_shift = random.randint(-5, 5)
            hor_shift = random.randint(-5, 5)
            x = np.roll(x, (ver_shift, hor_shift), axis=(0, 1))

        return x.astype(np.float32), label.astype(np.float32)

def create_dataloader(input, target, batch_size, train=True):
    """
    Create a data loader for the given input and target files.

    Args:
        input (str): Path to the input file.
        target (str): Path to the target file.
        batch_size (int): Batch size for the data loader.
        train (bool): Whether to create a data loader for training or evaluation.

    Returns:
        torch.utils.data.DataLoader: The created data loader.
    """
    # Load data from input and target files
    with open(input, 'rb') as f:
        xs = np.load(f)
    with open(target, 'rb') as f:
        labels = np.load(f)

    # Create OneStreamDataset instance
    dataset = OneStreamDataset(xs, labels, train=train, apply_random_shift=train)

    # Create DataLoader with the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_loader

def pred_input_function(xs, i, window_size=256, step_size=26):
    """
    Creates a minibatch of 50 500ms-spectrograms from a 2500ms-spectrogram using a sliding window
    with the window size of 500ms and the window shift of 50ms.

    This function takes a sequence of spectral segments (xs) and an index (i) representing
    the current position in the sequence. It extracts 50 elements, each of 500ms spectrogram with
    a shape of (257, 256), for the current position of a 2500ms-spectrogram with a shape of (257, 1299).

    The extraction process is done using a sliding window approach with a window size of 256 and a
    step size of 26 (the step size or window shift of 26 pixels is from floor(1299/50)).
    The first num_complete_elements can be extracted entirely from the current 2500ms-segment, while
    the remaining elements require concatenation with the next segment.

    A window shift of 26 pixels is around 50ms.
    26 * frame_shift = 26 * (92/48000*1000) = 49.833333ms

    # Got a spectrogram image of size 257x256 (freq, time) for a 500ms segment
    # A) To make num_freqs close to 256
    # nfft = 512 => num_freqs = 512/2 + 1 = 257 (1 for the origin, div by 2 for symmetricity of FFT)
    # B) To make num_frames close to 256
    # size = nfft = 512 (samples)
    # shift = (500/1000*48000 - 512) / (256-1) = 92.1 ~ 92,
    # where -1 in "(256 - 1)" means that having placed the last frame, calculate the distance between other adjoining frames
    #
    # Frame size = 512/48000*1000 = 10.7ms
    # Frame shift = 92/48000*1000 = 1.916ms
    # Frame size 10.7ms and shift 1.916ms with 82% ((10.7-1.916)/10.7) overlap
    # (500-10.7)/1.926 + 1 = 255.0498

    Args:
        xs (array-like): Input spectral segments, of shape (*, 257, 1299),
                         where * is number of batches (number of 2500ms-segments)
        i (int): The index of the current spectral segment.
        window_size (int): The size of each element (default: 256).
        step_size (int): The step size for sliding the window (default: 26).

    Returns:
        numpy.ndarray: A minibatch of input features, of shape (50, 257, 256).

    Note: The window shift of 50ms is used for generating the segment files with a resolution of 50ms
    for the test set.
    """
    num_elements = 50
    num_complete_elements = math.floor((xs.shape[2] - window_size) / step_size) + 1

    elements = []

    # Extract complete elements from the current spectral segment
    for j in range(num_complete_elements):
        start = j * step_size
        element = xs[i, :, start:start+window_size] # Take i-th batch with the whole frequency bins and a window time.
        elements.append(element)

    # Extract remaining elements that cross the segment boundary
    for k in range(num_elements - num_complete_elements):
        start = (num_complete_elements + k) * step_size
        init_element = xs[i, :, start:]
        remaining = window_size - init_element.shape[1]
        element = np.concatenate([init_element, xs[i+1, :, :remaining]], axis=1)
        elements.append(element)

    return np.array(elements, dtype=np.float32)

def train(model, optimizer, train_loader, dev_loader, num_epochs, save_epoch_interval, lr, lr_decay_interval, result_dir, mtest_loader=None):
    """
    Function for training the Vision Transformer (ViT) model using the training and evaluation/development data loaders.

    Args:
        model (ViT): The ViT model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        dev_loader (torch.utils.data.DataLoader): The data loader for the development set.
        num_epochs (int): Number of training epochs.
        save_epoch_interval (int): Interval for saving the model.
        lr (float): Learning rate.
        lr_decay_interval (int): Decay the learning rate every x epochs by lr *= 0.97
        result_dir (str): Path to the result directory to save the trained models.
        mtest_loader (torch.utils.data.DataLoader): The data loader for the test set monitoring (default: None)
    """
    # Whether to overwrite the result directory when it exists. Make a new directory when it does not exists.
    overwrite_result_directory(result_dir, args.overwrite)

    logger = init_logger(os.path.join(result_dir, "report.log"))  # logger for training
    logger.info(args)
    logger.info('Training new model')
    if args.pretrained_model: logger.info(f'Loading pretrained model from {pretrained_model}')


    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    best_dev_loss = sys.float_info.max
    best_dev_epoch = 0
    epoch = 0
    while epoch < num_epochs:
        accuracies = []
        train_loss = 0.0
        info_table = []
        model.train()  # Set the model to training mode

        # Iterate over the training data loader
        for batch_xs, batch_labels in tqdm.tqdm(train_loader, ascii=True, ncols=50):
            # Move the batch data to the device (GPU or CPU)
            batch_xs = batch_xs.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass through the model
            _, loss, accurs, _ = model(batch_xs, batch_labels)
            accuracies.extend([1 if val else 0 for val in accurs])
            train_loss += loss.item()

            # Backward pass and optimization step
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
            optimizer.step()

            # Check for NaN values in the loss
            if np.isnan(loss.item()):
                raise ValueError("NaN detected")

        # Compute average training loss and accuracy
        train_loss /= len(train_loader)
        train_accuracy = np.mean(accuracies)

        # Log the training accuracy and loss for the epoch
        info_table.append([epoch, "train_set", train_loss, train_accuracy])

        # Record training loss and accuracy in TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Evaluate on the development set
        model.eval()  # Set the model to evaluation mode
        accuracies = []
        eval_loss = 0.0

        # Disable gradient computation during evaluation
        with torch.no_grad():
            # Iterate over the development data loader
            for batch_xs, batch_labels in tqdm.tqdm(dev_loader, ascii=True, ncols=50):
                # Move the batch data to the device (GPU or CPU)
                batch_xs = batch_xs.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass through the model
                _, loss, accurs, _ = model(batch_xs, batch_labels)
                accuracies.extend([1 if val else 0 for val in accurs])
                eval_loss += loss.item()

        # Compute average evaluation loss and accuracy
        eval_loss /= len(dev_loader)
        eval_accuracy = np.mean(accuracies)

        if mtest_loader:
            # Evaluate on the test set for monitoring
            model.eval()  # Set the model to evaluation mode
            accuracies = []
            mtest_loss = 0.0

            # Disable gradient computation during test set monitoring
            with torch.no_grad():
                # Iterate over the test data loader for monitoring
                for batch_xs, batch_labels in tqdm.tqdm(mtest_loader, ascii=True, ncols=50):
                    # Move the batch data to the device (GPU or CPU)
                    batch_xs = batch_xs.to(device)
                    batch_labels = batch_labels.to(device)

                    # Forward pass through the model
                    _, loss, accurs, _ = model(batch_xs, batch_labels)
                    accuracies.extend([1 if val else 0 for val in accurs])
                    mtest_loss += loss.item()

            # Compute average evaluation loss and accuracy
            mtest_loss /= len(mtest_loader)
            mtest_accuracy = np.mean(accuracies)

        # Log the evaluation accuracy and loss
        info_table.append([epoch, "dev_set", eval_loss, eval_accuracy])
        if mtest_loader:
            info_table.append([epoch, "test_set", mtest_loss, mtest_accuracy])
            logger.info("\n" + tabulate.tabulate(info_table, headers=['epoch', 'dataset', 'loss', 'acc'], floatfmt='.4f', tablefmt='rst'))
        else:
            logger.info("\n" + tabulate.tabulate(info_table, headers=['epoch', 'dataset', 'loss', 'acc'], floatfmt='.4f', tablefmt='rst'))

        # Record evaluation loss and accuracy in TensorBoard
        writer.add_scalar('Loss/dev', eval_loss, epoch)
        writer.add_scalar('Accuracy/dev', eval_accuracy, epoch)
        if mtest_loader:
            writer.add_scalar('Loss/test', mtest_loss, epoch)
            writer.add_scalar('Accuracy/test', mtest_accuracy, epoch)

        # Decay the learning rate
        if epoch % lr_decay_interval == 0:
            logger.info(f"Decaying the learning rate from '{lr:.8f}' to '{lr * 0.97:.8f}'")
            lr *= 0.97
            for g in optimizer.param_groups:
                g['lr'] = lr

        # Save the checkpoints at specified intervals
        if epoch % save_epoch_interval == 0:
            # Save the trained model checkpoint
            model_path = os.path.join(result, f"model_e{epoch}.ckpt") # Save the trained model
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved: {model_path}")

        # Save the trained model checkpoint that has the lowest loss on the development set
        if best_dev_loss > eval_loss:
            best_dev_loss = eval_loss
            best_dev_epoch = epoch
            logger.info(f"Get a better dev loss {best_dev_loss:.3f} at epoch {best_dev_epoch} ... saving the model")
            model_path = os.path.join(result, "model_best_dev.ckpt") # Save the trained model
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved: {model_path}")

        # Save the latest trained model checkpoint
        model_path = os.path.join(result, "model.ckpt") # Save the trained model
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved: {model_path}")

        # Continue training
        if epoch == num_epochs - 1 and not args.exit:
            command = "python " + ' '.join([x for x in sys.argv])
            message = f"command: '{command}'\nresult: '{result_dir}'\n"
            continue_or_not, add_epochs = continue_train(message) # add 'python command' and 'result directory' to message
            if continue_or_not and add_epochs:
                num_epochs += add_epochs
                logging.info("Add {} more epochs".format(add_epochs))

        epoch += 1

    # Close the TensorBoard writer
    writer.close()

def predict(model, pred_data, model_path, avg_pred_win, batch_size):
    """
    Function for predicting with the trained Vision Transformer (ViT) model for the testing set.

    Args:
        model (ViT): The trained ViT model used for prediction.
        pred_data (str): Path to the file containing spectrogram arrays for prediction.
        model_path (str): Path to the saved model to be used for prediction.
        avg_pred_win (int): Size of the averaging window for predictions.
        batch_size (int): Minibatch size for prediction.

    Returns:
        numpy.ndarray: The predicted probabilities.
    """
    logger.info(args)

    # Load the input data
    with open(pred_data, 'rb') as f:
        predict_x = np.load(f)

    # Restore the saved model for prediction
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    logger.info(f'Model restored from {model_path}')

    # Perform predictions on the input data
    preds_list = []
    predictions = []
    logger.info('Predicting')
    model.eval()

    length = predict_x.shape[0]
    num_batches = length
    for i in range(num_batches-1):
        # Get the input batch using the pred_input_function
        inputs = pred_input_function(predict_x, i)
        # Run the model to get the predictions for the current batch
        preds, _ = model(torch.from_numpy(inputs).to(device))
        # Append the predictions to the preds_list
        preds_list.extend(preds.detach().cpu().numpy())
        if (i + 1) % 25 == 0:
            logger.info(f"Processed {i + 1}/{num_batches} batches")

    # Average predictions across consecutive windows
    for i in range(len(preds_list) - (avg_pred_win - 1)):
        # Calculate the mean predictions for the current window
        mean_preds = np.mean(preds_list[i:i + avg_pred_win], axis=0)
        # Append the mean predictions to the final predictions list
        predictions.append(mean_preds)

    logger.info(f"Predictions shape: {np.shape(predictions)}")
    logger.info(f"Used the model: {eval_model}")

    return np.array(predictions)

data_dir = "exp/data/division_sample_winmid0.05size0.5shift0.05_noisekeep5"
exp_dir = "exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/vit-run0/bs25lr0.0003evalinterval1avgpredwin5"
# Argument parser
parser = argparse.ArgumentParser(description=("Train or evaluate Vision Transformer (ViT)."))
# Data arguments
parser.add_argument("--train_input", type=str, default=data_dir+"/train_input.npy", help="Path to the input of the training set")
parser.add_argument("--train_target", type=str, default=data_dir+"/train_target.npy", help="Path to the target labels of the training set")
parser.add_argument("--dev_input", type=str, default=data_dir+"/dev_input.npy", help="Path to the input of the development set")
parser.add_argument("--dev_target", type=str, default=data_dir+"/dev_target.npy", help="Path to the target labels of the development set")
parser.add_argument("--mtest_input", type=str, default=data_dir+"/test_input.npy", help="Path to the input of the test set for monitoring")
parser.add_argument("--mtest_target", type=str, default=data_dir+"/test_target.npy", help="Path to the target labels of the test set for monitoring")
parser.add_argument("--test_input", type=str, default=data_dir+"/test_input_Athos.npy", help="Path to the input of the testing set")
parser.add_argument("--test_pred", type=str, default=exp_dir+"/eval/test_pred_Athos.npy", help="Path to save the predicted label probabilities for the test set")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the dataloader")
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id_marmoset.yaml", help="The YAML file that contains the label-to-labelID mapping.")

# Model arguments
parser.add_argument("--image_size", type=int, default=256, help="Image size (height, width)")
parser.add_argument("--patch_size", type=int, default=16, help="Patch size (height, width)")
parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
parser.add_argument("--heads", type=int, default= 6, help="Number of attention heads")
parser.add_argument("--mlp_dim", type=int, default=3072, help="Dimension of the MLP layer")
parser.add_argument("--pool", type=str, default='cls', choices=['cls', 'mean'], help="Pooling type")
parser.add_argument("--channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--dim_head", type=int, default=64, help="Dimension of each attention head")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--emb_dropout", type=float, default=0.1, help="Embedding dropout rate")
parser.add_argument("--return_attention", action='store_true', help="Return attention weights")
parser.add_argument("--return_logits", action='store_true', help="Return logits instead of probabilities")
parser.add_argument('--pretrained_model', default="", help="The path to pretrained model (model.ckpt). Example usage: python local/sandbox/riken_vit_train.py --pretrained_model conf/model/model.ckpt --label2id_yaml conf/dict/label2id.yaml")

# Optimizer arguments
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate of Adam optimizer")
parser.add_argument("--epsilon", type=float, default=0.001, help="Epsilon of Adam optimizer for numerical stability")

# Training arguments
parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--save_epoch_interval", type=int, default=2, help="Save the model every x epochs")
parser.add_argument("--lr_decay_interval", type=int, default=1, help="Decay the learning rate every x epochs by lr *= 0.97")

# Evaluation arguments
parser.add_argument("--eval_model", type=str, default=exp_dir+"/train/model.ckpt", help="Model path for prediction or evaluation") # pred
parser.add_argument("--avg_pred_win", type=int, default=5, help="Collect predicted probabilities by averaging across x consecutive predictions")

# Other arguments
parser.add_argument('--seed', type=int, default=2020, help='Seed')
parser.add_argument('--gpu', type=str, default='auto', # if 'auto', running three times in ipython will occupy three different GPUs.
                    help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least GPU memory")
parser.add_argument("--result", type=str, default=exp_dir+"/train/", help="Result directory")
parser.add_argument('--overwrite', action='store_true', help='overwrite the result')
parser.add_argument('--exit', action='store_true', help="Immediately exit training or continue with additional epochs")

# Parse arguments
args = parser.parse_args()
print(args)

# Assign arguments to variables
# Data
train_input = args.train_input
train_target = args.train_target
dev_input = args.dev_input
dev_target = args.dev_target
mtest_input = args.mtest_input
mtest_target = args.mtest_target
test_input = args.test_input
test_pred = args.test_pred
batch_size = args.batch_size
label2id = OmegaConf.load(args.label2id_yaml)
# Model
dropout = args.dropout
pretrained_model = args.pretrained_model
# Optimizer
lr = args.lr
epsilon = args.epsilon
# Training
num_epochs = args.num_epochs
save_epoch_interval = args.save_epoch_interval
lr_decay_interval = args.lr_decay_interval
# Testing
eval_model = args.eval_model
avg_pred_win = args.avg_pred_win
# Others
result = args.result
overwrite = args.overwrite

set_seed(args.seed)
device = set_device(args.gpu)
print(f"Device: {device}")

# Create and initialize the ViT model
model = ViT(
    image_size=args.image_size,
    patch_size=args.patch_size,
    num_classes=len(label2id),
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    pool=args.pool,
    channels=args.channels,
    dim_head=args.dim_head,
    dropout=args.dropout,
    emb_dropout=args.emb_dropout,
    return_attention=args.return_attention,
    return_logits=args.return_logits
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)
if args.pretrained_model:
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device)["model"])
    optimizer.load_state_dict(torch.load(args.pretrained_model, map_location=device)["optimizer"])

# # Training stage
# start_time = datetime.datetime.now()
# train_loader = create_dataloader(train_input, train_target, batch_size, train=True)
# dev_loader = create_dataloader(dev_input, dev_target, batch_size, train=False)
# mtest_loader = create_dataloader(mtest_input, mtest_target, batch_size, train=False) if mtest_input and mtest_target else None # test set for monitoring
# train(model, optimizer, train_loader, dev_loader, num_epochs, save_epoch_interval, lr, lr_decay_interval, result, mtest_loader)
# duration = datetime.datetime.now() - start_time
# print(f'Time taken to complete the training: {duration.seconds // 3600:02}:{(duration.seconds // 60) % 60:02}:{duration.seconds % 60:02}')

# Prediction stage
if not os.path.exists(os.path.dirname(test_pred)): os.makedirs(os.path.dirname(test_pred))
logger = init_logger(os.path.join(os.path.dirname(test_pred), "report.log")) # logger for testing
predictions = predict(model, test_input, model_path=eval_model, avg_pred_win=avg_pred_win, batch_size=batch_size)
np.save(test_pred, predictions)
