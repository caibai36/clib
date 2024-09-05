# Implemented by bin-wu at 02:52 on 10 April 2024 from the MIT's implementation
#    -> v1: Separate the training and the testing codes.
#           Add a logger.
#    -> v2: Change interations to epochs
#           Add a dataloader for training and development sets
#    -> v3: Replace CNN with ViT
#       v3.0.1: Save the trained model checkpoint that has the highest accuarcy on the development set (model_best_dev.ckpt)


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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

truth_values=['cha','chi','ek','ph','ts','tr','trph','tw','noise']

# Argument parser
parser = argparse.ArgumentParser(description=("Train or evaluate MIT ViT 72. (Reference: 'new_train_72.py' from https://marmosetbehavior.mit.edu/, supporting TensorFlow 2)"))

# Data arguments
parser.add_argument("--train_input1", type=str, default="exp/data/mit_sample/train_input1", help="Path to the first stream input of the training set")
parser.add_argument("--train_input2", type=str, default="exp/data/mit_sample/train_input2", help="Path to the second stream input of the training set")
parser.add_argument("--train_target_single1", type=str, default="exp/data/mit_sample/train_target_single1", help="Path to the first stream target labels of the training set")
parser.add_argument("--train_target_single2", type=str, default="exp/data/mit_sample/train_target_single2", help="Path to the second stream target labels of the training set")
parser.add_argument("--dev_input1", type=str, default="exp/data/mit_sample/dev_input1", help="Path to the first stream input of the development set")
parser.add_argument("--dev_input2", type=str, default="exp/data/mit_sample/dev_input2", help="Path to the second stream input of the development set")
parser.add_argument("--dev_target_single1", type=str, default="exp/data/mit_sample/dev_target_single1", help="Path to the first stream target labels of the development set")
parser.add_argument("--dev_target_single2", type=str, default="exp/data/mit_sample/dev_target_single2", help="Path to the second stream target labels of the development set")
parser.add_argument("--test_input1", type=str, default="exp/data/mit_sample/test_input1_Athos", help="Path to the first stream input of the testing set")
parser.add_argument("--test_input2", type=str, default="exp/data/mit_sample/test_input2_Porthos", help="Path to the second stream input of the testing set")
parser.add_argument("--test_pred1", type=str, default="exp/run/mit_sample/mit_sample0/mit_vit_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos", help="Path to save the predicted label probabilities of the first stream for the test set")
parser.add_argument("--test_pred2", type=str, default="exp/run/mit_sample/mit_sample0/mit_vit_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos", help="Path to save the predicted label probabilities of the second stream for the test set")

# Model arguments
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the dataloader")
parser.add_argument("--dropout_rate", type=float, default=0.4, help="Drop rate of dropout layer for shared layers")
parser.add_argument("--eval_model", type=str, default="exp/run/mit_sample/mit_sample0/mit_vit_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/model.ckpt", help="Model path for prediction or evaluation")
parser.add_argument("--shared_dim", type=int, default=1024, help="Dimension of the shared fully connected layer")

# ViT specific arguments
parser.add_argument("--image_size", type=int, default=256, help="Size of input image")
parser.add_argument("--patch_size", type=int, default=16, help="Size of patch")
parser.add_argument("--num_classes", type=int, default=9, help="Number of classes")
parser.add_argument("--dim", type=int, default=384, help="Dimension of ViT")
parser.add_argument("--depth", type=int, default=12, help="Depth of ViT")
parser.add_argument("--heads", type=int, default=6, help="Number of attention heads")
parser.add_argument("--mlp_dim", type=int, default=3072, help="Dimension of MLP layer")
parser.add_argument("--pool", type=str, default='cls', help="Pooling type (cls or mean)")
parser.add_argument("--channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--dim_head", type=int, default=64, help="Dimension of each attention head")
parser.add_argument("--vit_dropout", type=float, default=0.1, help="Drop rate of dropout layer for ViT")
parser.add_argument("--emb_dropout", type=float, default=0.1, help="Embedding dropout rate")
parser.add_argument("--return_attention", action='store_true', help="Whether to return attention weights")

# Optimizer arguments
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate of Adam optimizer")
parser.add_argument("--epsilon", type=float, default=0.001, help="Epsilon of Adam optimizer for numerical stability")

# Training arguments
parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--eval_epoch_interval", type=int, default=1, help="Save the model every x epochs")

# Evaluation arguments
parser.add_argument("--avg_pred_win", type=int, default=5, help="Collect predicted probabilities by averaging across x consecutive predictions")

# Other arguments
parser.add_argument('--seed', type=int, default=2020, help='Seed')
parser.add_argument('--gpu', type=str, default='auto', help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least GPU memory")
parser.add_argument("--result", type=str, default="exp/run/mit_sample/mit_sample0/mit_vit_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/", help="Result directory")
parser.add_argument('--overwrite', action='store_true', help='overwrite the result')
parser.add_argument('--exit', action='store_true', help="Immediately exit training or continue with additional epochs")

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
# Training
# Data
train_input1 = args.train_input1
train_input2 = args.train_input2
train_target_single1 = args.train_target_single1
train_target_single2 = args.train_target_single2
dev_input1 = args.dev_input1
dev_input2 = args.dev_input2
dev_target_single1 = args.dev_target_single1
dev_target_single2 = args.dev_target_single2
batch_size = args.batch_size
# Model
dropout_rate = args.dropout_rate
vit_dropout = args.vit_dropout
shared_dim = args.shared_dim
# Optimizer
lr = args.lr
epsilon = args.epsilon
# Training
num_epochs = args.num_epochs
eval_epoch_interval = args.eval_epoch_interval
# Others
result = args.result
overwrite = args.overwrite

# Testing
test_input1 = args.test_input1
test_input2 = args.test_input2
test_pred1 = args.test_pred1
test_pred2 = args.test_pred2
eval_model = args.eval_model
avg_pred_win = args.avg_pred_win

model_path = os.path.join(result, "model.ckpt") # Save the training model

def set_seed(seed):
    """
    Set the seed for reproducibility across NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

set_seed(args.seed)
device = set_device(args.gpu)
print(f"Device: {device}")

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

class TwoStreamViTModel(nn.Module):
    """
    A two-stream Vision Transformer model.

    Args:
        vit_params (dict): Parameters for the ViT model.
        dropout_rate (float): The dropout rate for the shared layers.
        shared_dim (int): The dimension of the shared fully connected layer.
    """

    def __init__(self, vit_params, dropout_rate, shared_dim):
        super(TwoStreamViTModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.shared_dim = shared_dim

        # First ViT stream
        self.vit1 = ViT(**vit_params)

        # Second ViT stream
        self.vit2 = ViT(**vit_params)

        # Shared fully connected layers
        vit_output_dim = vit_params['dim']  # Assuming the ViT output dimension is equal to 'dim'
        self.fc1 = nn.Linear(vit_output_dim * 2, self.shared_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(self.shared_dim, 9)
        self.fc3 = nn.Linear(self.shared_dim, 9)

    def forward(self, x1, x2, y1=None, y2=None, mode='train'):
        """
        Forward pass of the model.

        Args:
            x1 (torch.Tensor): Input tensor for the first stream, of shape (batch_size, 257, 256).
            x2 (torch.Tensor): Input tensor for the second stream, of shape (batch_size, 257, 256).
            y1 (torch.Tensor, optional): Ground truth labels for the first output, of shape (batch_size,). Defaults to None.
            y2 (torch.Tensor, optional): Ground truth labels for the second output, of shape (batch_size,). Defaults to None.
            mode (str, optional): The mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            If y1 and y2 are provided:
                tuple: A tuple containing:
                    - probs1 (torch.Tensor): Probabilities for the first output, of shape (batch_size, 9).
                    - probs2 (torch.Tensor): Probabilities for the second output, of shape (batch_size, 9).
                    - loss (torch.Tensor): The computed loss.
                    - accuracy (list): A list of accuracies for each sample in the batch.
                    - attention_weights (list): A list of attention weight tensors.
            If y1 and y2 are not provided:
                tuple: A tuple containing:
                    - probs1 (torch.Tensor): Probabilities for the first output, of shape (batch_size, 9).
                    - probs2 (torch.Tensor): Probabilities for the second output, of shape (batch_size, 9).
                    - attention_weights (list): A list of attention weight tensors.
        """
        # Pass input x1 through the first ViT stream
        x1, attn1 = self.vit1(x1)

        # Pass input x2 through the second ViT stream
        x2, attn2 = self.vit2(x2)

        # Concatenate the outputs of the two ViT streams
        x = torch.cat((x1, x2), dim=1)

        # Pass the concatenated features through the shared fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if mode == 'train' else x

        # Obtain the logits for the two outputs
        logits1 = self.fc2(x)
        logits2 = self.fc3(x)

        # Compute the probabilities using softmax
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        attention_weights = [attn1, attn2]

        # If ground truth labels are provided, compute loss and accuracy
        if y1 is not None and y2 is not None:
            loss1 = F.cross_entropy(logits1, y1)
            loss2 = F.cross_entropy(logits2, y2)
            loss = loss1 + loss2

            classes1 = logits1.argmax(dim=1)
            classes2 = logits2.argmax(dim=1)
            accuracy = ((classes1 == y1) & (classes2 == y2)).float()

            return probs1, probs2, loss, accuracy.tolist(), attention_weights

        # If ground truth labels are not provided, return only the probabilities and attention weights
        return probs1, probs2, attention_weights

class TwoStreamCNNModel(nn.Module):
    """
    A two-stream convolutional neural network model.

    Args:
        dropout_rate (float, optional): The dropout rate. Defaults to 0.5.
    """

    def __init__(self, dropout_rate=0.5):
        super(TwoStreamCNNModel, self).__init__()
        self.dropout_rate = dropout_rate

        # First convolutional stream
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second convolutional stream
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(8 * 8 * 64 * 2, 1024)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(1024, 9)
        self.fc3 = nn.Linear(1024, 9)

    def forward(self, x1, x2, y1=None, y2=None, mode='train'):
        """
        Forward pass of the model.

        Args:
            x1 (torch.Tensor): Input tensor for the first stream, of shape (batch_size, 257, 256).
            x2 (torch.Tensor): Input tensor for the second stream, of shape (batch_size, 257, 256).
            y1 (torch.Tensor, optional): Ground truth labels for the first output, of shape (batch_size,). Defaults to None.
            y2 (torch.Tensor, optional): Ground truth labels for the second output, of shape (batch_size,). Defaults to None.
            mode (str, optional): The mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            If y1 and y2 are provided:
                tuple: A tuple containing:
                    - probs1 (torch.Tensor): Probabilities for the first output, of shape (batch_size, 9).
                    - probs2 (torch.Tensor): Probabilities for the second output, of shape (batch_size, 9).
                    - loss (torch.Tensor): The computed loss.
                    - accuracy (list): A list of accuracies for each sample in the batch.
            If y1 and y2 are not provided:
                tuple: A tuple containing:
                    - probs1 (torch.Tensor): Probabilities for the first output, of shape (batch_size, 9).
                    - probs2 (torch.Tensor): Probabilities for the second output, of shape (batch_size, 9).
        """
        # Pass input x1 through the first convolutional stream
        x1 = x1.unsqueeze(1)  # Add channel dimension
        x1 = self.conv1(x1)
        x1 = x1.view(x1.size(0), -1)

        # Pass input x2 through the second convolutional stream
        x2 = x2.unsqueeze(1)  # Add channel dimension
        x2 = self.conv2(x2)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate the outputs of the two convolutional streams
        x = torch.cat((x1, x2), dim=1)

        # Pass the concatenated features through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if mode == 'train' else x

        # Obtain the logits for the two outputs
        logits1 = self.fc2(x)
        logits2 = self.fc3(x)

        # Compute the probabilities using softmax
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        # If ground truth labels are provided, compute loss and accuracy
        if y1 is not None and y2 is not None:
            loss1 = F.cross_entropy(logits1, y1)
            loss2 = F.cross_entropy(logits2, y2)
            loss = loss1 + loss2

            classes1 = logits1.argmax(dim=1)
            classes2 = logits2.argmax(dim=1)
            accuracy = ((classes1 == y1) & (classes2 == y2)).float()

            return probs1, probs2, loss, accuracy.tolist()

        # If ground truth labels are not provided, return only the probabilities
        return probs1, probs2

class TwoStreamDataset(Dataset):
    """
    A PyTorch Dataset class for the two-stream data.

    Args:
        xs1 (numpy.ndarray): Input data for the first stream, of shape (*, 257, 256).
        xs2 (numpy.ndarray): Input data for the second stream, of shape (*, 257, 256).
        labels1 (numpy.ndarray): Labels corresponding to xs1, of shape (*, num_classes).
        labels2 (numpy.ndarray): Labels corresponding to xs2, of shape (*, num_classes).
        train (bool): Whether the dataset is used for training or evaluation/development.
        switch_streams (bool): Whether to randomly switch the input-label pairing. Default is True.
        apply_random_shift (bool): Whether to apply random shifts to each sample. Default is True.

    Note:
        Two optional randomizations:
        For the two-stream system, randomly feed data in two cases:
        data1 to stream1 and data2 to stream2 or
        data1 to stream2 and data2 to stream1,
        where each data include its input and label.

        Each sample in the batch applies a new random shift within 5 pixels.
    """

    def __init__(self, xs1, xs2, labels1, labels2, train=True, switch_streams=True, apply_random_shift=True):
        self.xs1 = xs1
        self.xs2 = xs2
        self.labels1 = labels1
        self.labels2 = labels2
        self.train = train
        self.switch_streams = switch_streams
        self.apply_random_shift = apply_random_shift

    def __len__(self):
        return len(self.xs1)

    def __getitem__(self, idx):
        x1 = self.xs1[idx]
        x2 = self.xs2[idx]
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]

        if self.train:
            # Randomly choose the input-label pairing if switch_streams is True
            if self.switch_streams and random.random() < 0.5:
                x1, x2 = x2, x1
                label1, label2 = label2, label1

            # Randomly roll each sample independently if apply_random_shift is True
            if self.apply_random_shift:
                ver_shift = random.randint(-5, 5)
                hor_shift = random.randint(-5, 5)
                x1 = np.roll(x1, (ver_shift, hor_shift), axis=(0, 1))
                x2 = np.roll(x2, (ver_shift, hor_shift), axis=(0, 1))


        return x1.astype(np.float32), x2.astype(np.float32), \
            label1.astype(np.float32), label2.astype(np.float32)

def create_dataloader(input1, input2, target1, target2, batch_size, train=True):
    """
    Create a data loader for the given input and target files.

    Args:
        input1 (str): Path to the first input file.
        input2 (str): Path to the second input file.
        target1 (str): Path to the first target file.
        target2 (str): Path to the second target file.
        batch_size (int): Batch size for the data loader.
        train (bool): Whether to create a data loader for training or evaluation.

    Returns:
        torch.utils.data.DataLoader: The created data loader.
    """
    # Load data from input and target files
    with open(input1, 'rb') as f:
        xs1 = np.load(f)
    with open(input2, 'rb') as f:
        xs2 = np.load(f)
    with open(target1, 'rb') as f:
        labels1 = np.load(f)
    with open(target2, 'rb') as f:
        labels2 = np.load(f)

    # Create TwoStreamDataset instance
    dataset = TwoStreamDataset(xs1, xs2, labels1, labels2, train=train, switch_streams=train, apply_random_shift=train)

    # Create DataLoader with the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_loader

def pred_input_function(xs1, xs2, i, window_size=256, step_size=26):
    """
    Creates a minibatch of 50 500ms-spectrograms from an 2500ms-spectrogram using a sliding window
    with the window size of 500ms and the window shift of 50ms.

    This function takes two sequences of spectral segments (xs1 and xs2) and an index (i) representing
    the current position in the sequences. It extracts 50 elements, each of 500ms spectrogram with
    a shape of (257, 256), for the current position of a 2500ms-spectrogram with a shape of (257, 1299).

    The extraction process is done using a sliding window approach with a window size of 256 and a
    step size of 26 (the step size or window shift of 26 pixel is from (floor(1299/50)).
    The first num_complete_elements can be extracted entirely from the current 2500ms-segment, while
    the remaining elements require concatenation with the next segment.


    Args:
        xs1 (array-like): Input spectral segments for the first stream, of shape (*, 257, 1299).
        xs2 (array-like): Input spectral segments for the second stream, of shape (*, 257, 1299).
        i (int): The index of the current spectral segment.
        window_size (int): The size of each element (default: 256).
        step_size (int): The step size for sliding the window (default: 26).

    Returns:
        dict: A dictionary containing the i-th minibatch of input features.
              Keys: 'x', 'x2'. Values: NumPy arrays of shape (50, 257, 256) for each.
              The batch_size is fixed as 50.

    Note: The window shift of 50ms is used for generating the segment files with resolution of 50ms
    for the test set.
    """
    num_elements = 50
    num_complete_elements = math.floor((xs1.shape[2] - window_size) / step_size) + 1

    new_features = {}
    for key, xs in zip(['x', 'x2'], [xs1, xs2]):
        elements = []

        # Extract complete elements from the current spectral segment
        for j in range(num_complete_elements):
            start = j * step_size
            element = xs[i, :, start:start+window_size]
            elements.append(element)

        # Extract remaining elements that cross the segment boundary
        for k in range(num_elements - num_complete_elements):
            start = (num_complete_elements + k) * step_size
            init_element = xs[i, :, start:]
            remaining = window_size - init_element.shape[1]
            element = np.concatenate([init_element, xs[i+1, :, :remaining]], axis=1)
            elements.append(element)

        new_features[key] = np.array(elements, dtype=np.float32)

    return new_features

def train(model, optimizer, train_loader, dev_loader, num_epochs, eval_epoch_interval, lr, model_path):
    """
    Function for training the network using the training and evaluation/development data loaders.

    Args:
        model (TwoStreamViTModel): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        dev_loader (torch.utils.data.DataLoader): The data loader for the development set.
        num_epochs (int): Number of training epochs.
        eval_epoch_interval (int): Interval for saving the model.
        lr (float): Learning rate.
        model_path (str): Path to save the trained model.
    """
    # Initialize the directory of path to save the model
    result_dir = os.path.dirname(model_path)
    # Whether to overwrite the result directory when it exists. Make a new directory when it does not exists.
    overwrite_result_directory(result_dir, args.overwrite)

    logger = init_logger(os.path.join(os.path.dirname(model_path), "report.log"))  # logger for training
    logger.info(args)
    logger.info('Training new model')

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    best_dev_acc = 0
    best_dev_epoch = 0

    epoch = 0
    while epoch < num_epochs:
        accuracies = []
        train_loss = 0.0
        info_table = []
        model.train()  # Set the model to training mode

        # Iterate over the training data loader
        for batch_xs1, batch_xs2, batch_labels1, batch_labels2 in tqdm.tqdm(train_loader, ascii=True, ncols=50):
            # Move the batch data to the device (GPU or CPU)
            batch_xs1 = batch_xs1.to(device)
            batch_xs2 = batch_xs2.to(device)
            batch_labels1 = batch_labels1.argmax(dim=1).to(device)
            batch_labels2 = batch_labels2.argmax(dim=1).to(device)

            # Forward pass through the model
            _, _, loss, accurs, _ = model(batch_xs1, batch_xs2, batch_labels1, batch_labels2)
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
        # logger.info(f"Epoch {epoch}/{num_epochs} Train loss: {train_loss:.4f} Train accuracy: {train_accuracy:.4f}")
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
            for batch_xs1, batch_xs2, batch_labels1, batch_labels2 in tqdm.tqdm(dev_loader, ascii=True, ncols=50):
                # Move the batch data to the device (GPU or CPU)
                batch_xs1 = batch_xs1.to(device)
                batch_xs2 = batch_xs2.to(device)
                batch_labels1 = batch_labels1.argmax(dim=1).to(device)
                batch_labels2 = batch_labels2.argmax(dim=1).to(device)

                # Forward pass through the model
                _, _, loss, accurs, _ = model(batch_xs1, batch_xs2, batch_labels1, batch_labels2)
                accuracies.extend([1 if val else 0 for val in accurs])
                eval_loss += loss.item()

        # Compute average evaluation loss and accuracy
        eval_loss /= len(dev_loader)
        eval_accuracy = np.mean(accuracies)

        # Log the evaluation accuracy and loss
        # logger.info(f"Eval loss: {eval_loss:.4f} Eval accuracy: {eval_accuracy:.4f}")
        info_table.append([epoch, "dev_set", eval_loss, eval_accuracy])
        logger.info("\n" + tabulate.tabulate(info_table, headers=['epoch', 'dataset', 'loss', 'acc'], floatfmt='.4f', tablefmt='rst'))

        # Record evaluation loss and accuracy in TensorBoard
        writer.add_scalar('Loss/dev', eval_loss, epoch)
        writer.add_scalar('Accuracy/dev', eval_accuracy, epoch)

        # Save the checkpoints at specified intervals
        if epoch % eval_epoch_interval == 0:
            # Decay the learning rate
            logger.info(f"Decaying the learning rate from '{lr:.8f}' to '{lr * 0.97:.8f}'")
            lr *= 0.97
            for g in optimizer.param_groups:
                g['lr'] = lr

            # Save the trained model checkpoint
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved: {model_path}")

            epoch_model_path = os.path.join(result_dir, f"model_e{epoch}.ckpt")
            torch.save(checkpoint, epoch_model_path)

        # Save the trained model checkpoint that has the highest accuarcy on the development set
        if eval_accuracy > best_dev_acc:
            best_dev_acc = eval_accuracy
            best_dev_epoch = epoch
            logger.info(f"New best dev accuracy {best_dev_acc:.4f} at epoch {best_dev_epoch}")

            # Save the best model checkpoint
            best_model_path = os.path.join(result_dir, "model_best_dev.ckpt")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, best_model_path)
            logger.info(f"Best model saved: {best_model_path}")

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

def predict(model, pred_data1, pred_data2, predictions_file1, predictions_file2, model_path, avg_pred_win, batch_size):
    """
    Function for predicting with the trained network for the testing set.

    Args:
        model (TwoStreamViTModel): The trained model used for prediction.
        pred_data1 (str): Path to the first file containing spectrogram arrays for prediction.
        pred_data2 (str or None): Path to the second input file from the same session.
                                  If None, an array of zeros will be used as the second input.
        predictions_file1 (str): Path to save the predicted probabilities for the first input.
        predictions_file2 (str): Path to save the predicted probabilities for the second input.
        model_path (str): Path to the saved model to be used for prediction.
        avg_pred_win (int): Size of the averaging window for predictions.
        batch_size (int): Minibatch size for prediction.

    Returns:
        None
    """
    # Make output directories for predicted files
    for output_dir in [os.path.dirname(predictions_file1), os.path.dirname(predictions_file2)]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    logger = init_logger(os.path.join(os.path.dirname(predictions_file1), "report.log")) # logger for testing
    logger.info(args)

    # Load the first input data
    with open(pred_data1, 'rb') as f:
        predict_x1 = np.load(f)

    # Load the second input data or create an array of zeros if pred_data2 is None
    if pred_data2 is not None:
        with open(pred_data2, 'rb') as f:
            predict_x2 = np.load(f)
    else:
        predict_x2 = np.zeros_like(predict_x1, dtype=np.float16)

    # Restore the saved model for prediction
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    logger.info(f'Model restored from {model_path}')

    # Perform predictions on the input data
    preds_list = []
    preds_list2 = []
    predictions = []
    predictions2 = []
    logger.info('Predicting')
    model.train(False)

    length = min(predict_x1.shape[0], predict_x2.shape[0])
    num_batches = length
    for i in range(num_batches-1):
        # Get the input batch using the pred_input_function
        inputs = pred_input_function(predict_x1, predict_x2, i)
        # Run the model to get the predictions for the current batch
        preds, preds2, _ = model(torch.from_numpy(inputs['x']).to(device),
                              torch.from_numpy(inputs['x2']).to(device))
        # Append the predictions to the preds_list and preds_list2
        preds_list.extend(preds.detach().cpu().numpy())
        preds_list2.extend(preds2.detach().cpu().numpy())
        if (i + 1) % 25 == 0:
            logger.info(f"Processed {i + 1}/{num_batches} batches")

    # Average predictions across consecutive windows
    for i in range(len(preds_list) - (avg_pred_win - 1)):
        # Calculate the mean predictions for the current window
        mean_preds = np.mean(preds_list[i:i + avg_pred_win], axis=0)
        mean_preds2 = np.mean(preds_list2[i:i + avg_pred_win], axis=0)
        # Append the mean predictions to the final predictions lists
        predictions.append(mean_preds)
        predictions2.append(mean_preds2)

    logger.info(f"Predictions shape: {np.shape(predictions)}")
    logger.info(f"Used the model: {eval_model}")

    # Save the predictions to files
    np.save(predictions_file1, predictions)
    np.save(predictions_file2, predictions2)

if __name__ == '__main__':
    vit_params = {
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'num_classes': args.num_classes,
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'mlp_dim': args.mlp_dim,
        'pool': args.pool,
        'channels': args.channels,
        'dim_head': args.dim_head,
        'dropout': args.vit_dropout,
        'emb_dropout': args.emb_dropout,
        'return_attention': args.return_attention,
        'return_latent': True
    }

    model = TwoStreamViTModel(vit_params, dropout_rate=args.dropout_rate, shared_dim=args.shared_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)

    # # Create data loaders for training and development sets
    # start_time = datetime.datetime.now()
    # train_loader = create_dataloader(train_input1, train_input2, train_target_single1, train_target_single2, batch_size, train=True)
    # dev_loader = create_dataloader(dev_input1, dev_input2, dev_target_single1, dev_target_single2, batch_size, train=False)
    # train(model, optimizer, train_loader, dev_loader, num_epochs, eval_epoch_interval, lr, model_path)
    # duration = datetime.datetime.now() - start_time
    # print(f'Time taken to complete the training: {duration.seconds // 3600:02}:{(duration.seconds // 60) % 60:02}:{duration.seconds % 60:02}')

    set_seed(args.seed)
    predict(model, test_input1, test_input2, test_pred1, test_pred2, model_path=eval_model, avg_pred_win=avg_pred_win, batch_size=batch_size)
