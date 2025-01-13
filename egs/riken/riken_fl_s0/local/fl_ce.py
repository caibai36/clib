# Implemented by bin-wu at 16:57 on 7 January 2025

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

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F

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

class PhonemeDataset(Dataset):
    """
    Dataset for syllable sequences with phoneme or syllable-level forward length computation.

    Input is always in syllable format where phonemes within syllables are connected by a
    connector character (default: underscore). The syllable_fl parameter controls whether to
    compute forward length at syllable level or phoneme level.

    Args:
        data_path (str): Path to corpus file containing syllable sequences, one per line
        label2id (dict): Dictionary mapping tokens to integer IDs, must include special
                        tokens '<mask>', '<pad>', and '<unk>'
        max_seq_length (int, optional): Maximum length for sequences, longer sequences
                                      will be truncated. Defaults to 512.
        syllable_fl (bool, optional): If True, compute forward length using syllables.
                                    If False, compute using phonemes. Defaults to False.
        connector (str, optional): Character used to connect phonemes in syllables.
                                 Defaults to '_'.

    Example input format (always syllables):
        c_ai f_u g_u sh_iii h_ui
        c_ai f_u x_in g_uan n_ian

    When syllable_fl=True:
        Treats each syllable as a unit: ['c_ai', 'f_u', 'g_u', 'sh_iii', 'h_ui']

    When syllable_fl=False:
        Splits syllables into phonemes: ['c', 'ai', 'f', 'u', 'g', 'u', 'sh', 'iii', 'h', 'ui']
    """
    def __init__(self, data_path, label2id, max_seq_length=512, syllable_fl=False, connector='_'):
        # Store configuration
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.connector = connector
        self.syllable_fl = syllable_fl

        # Get special token IDs from vocabulary
        self.pad_id = label2id['<pad>']    # For padding shorter sequences

        # Load and process sequences from file
        with open(data_path, 'r') as f:
            self.sequences = []
            for line_idx, line in enumerate(f, 1):
                original_line = line.strip()
                syllables = original_line.split()

                if syllable_fl:
                    # For syllable-level forward length: keep syllables intact
                    tokens = syllables
                else:
                    # For phoneme-level forward length: split syllables into phonemes
                    tokens = [phoneme
                            for syllable in syllables
                            for phoneme in syllable.split(self.connector)]

                # Check for unknown tokens and convert to ids
                unknowns = [t for t in tokens if t not in label2id]
                if unknowns:
                    print(f"Warning: Unknown tokens in {data_path} at line {line_idx}: {unknowns}")
                    print(f"Utterance: {original_line}")

                # Convert tokens to ids, using <unk> for OOV tokens
                ids = [label2id.get(token, label2id['<unk>']) for token in tokens]

                # Only add non-empty sequences
                if len(ids) > 0:
                    self.sequences.append(ids)

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single processed sequence from the dataset.

        The sequence is truncated to max_seq_length when too long

        Args:
            idx (int): Index of sequence to retrieve

        Returns:
            dict: Contains:
                - input_ids: Tensor of token ids [max_seq_length]
                - labels: Original unmasked ids [max_seq_length]
                - pad_id: The padding token ID
        """
        # Get sequence and truncate if needed
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_length:
            print(f"Warning: Sequence at index {idx} exceeds max_seq_length ({len(seq)} > {self.max_seq_length}). Truncating.")
            seq = seq[:self.max_seq_length]

        # Convert to tensors without padding
        input_ids = torch.tensor(seq)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pad_id': self.pad_id  # Pass pad_id for collate_fn
        }

def collate_fn(batch):
    """
    Collate function for DataLoader that pads sequences to the maximum length in the current batch.
    This enables more efficient processing by only padding to the necessary length for each batch,
    rather than padding all sequences to the global max_seq_length.

    Args:
        batch: List of dictionaries from PhonemeDataset, where each dictionary contains:
            - input_ids (torch.Tensor): Unpadded token IDs [seq_len]
            - labels (torch.Tensor): Unpadded label IDs [seq_len]
            - pad_id (int): The padding token ID

    Returns:
        dict: A dictionary containing batched and padded tensors:
            - input_ids (torch.Tensor): Padded input tensors [batch_size, max_batch_len]
            - labels (torch.Tensor): Padded label tensors [batch_size, max_batch_len]
            - attention_mask (torch.Tensor): Binary mask for valid positions [batch_size, max_batch_len]
                                          (1 for tokens, 0 for padding)

    Note:
        Sequences longer than max_seq_length are already truncated in PhonemeDataset.__getitem__,
        so this function only needs to handle padding up to the longest sequence in the batch.
    """
    # Find max length in this batch
    max_len = max(len(x['input_ids']) for x in batch)

    # Get padding token id
    pad_id = batch[0]['pad_id']

    # Pad sequences to max length in batch
    # Add 0 padding elements to the left of the sequence
    # Add (max_len - len(x['input_ids'])) padding elements to the right.
    input_ids = [F.pad(x['input_ids'], (0, max_len - len(x['input_ids'])), value=pad_id)
                for x in batch]
    labels = [F.pad(x['labels'], (0, max_len - len(x['labels'])), value=pad_id)
             for x in batch]

    # Stack tensors
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = (input_ids != pad_id)  # 1 for tokens, 0 for padding

    return {
        'input_ids': input_ids,      # Padded token ids
        'labels': labels,            # Padded label ids
        'pad_id': pad_id,
        'attention_mask': attention_mask  # Mask for valid positions
    }

# The code adapted from https://github.com/lucidrains/vit-pytorch
#
# Adapted by bin-wu on 2024/01/06 at 23:53:
# Adapted from vision transformer (ViT) to NLP transformer
# - Implement bert based on transformer
# - Add support of attention mask w.r.t. sequence padding for NLP
# - Added comments and docstrings
# - Implemented returning attention weights
# - Enhanced the forward method to handle labels, among other improvements

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each attention head. Defaults to 64.
        dropout (float, optional): Dropout probability. Defaults to 0.
        return_attention (bool, optional): Whether to return attention weights. Defaults to False.

    Shape:
        - Input: (batch_size, seq_len, dim)
        - Output: (batch_size, seq_len, dim), attention weights (optional)
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., return_attention=False):
        super().__init__()
        inner_dim = dim_head * heads # d_model of transformer
        project_out = not (heads == 1 and dim_head == dim)

        self.return_attention = return_attention
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Single linear layer to compute query, key, and value
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attention_mask=None):
        """Forward pass of attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim]
            attention_mask (torch.Tensor, optional): Boolean mask for attention
                [batch_size, seq_len]. Defaults to None.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape [batch_size, seq_len, dim]
                - torch.Tensor or None: Attention weights if return_attention=True
        """
        # Apply layer normalization
        x = self.norm(x) # pre_norm with shape: b n input_dim

        # Project input into query, key, and value representations with different transformation
        # Rearrange q, k, v to separate the heads
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 3 b n (h d)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # d_model=hxd=headxhead_dim; n=hxw=num_patches_heightxnum_patches_width

        # Compute scaled dot-product attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b h n d x b h d n -> b h n n

        # Apply attention mask if provided
        # Typical approach (only mask out invalid key positions)
        # This means: For each query position i, do not attend to masked positions j.
        # But a masked query can still produce logits (usually we just let them be
        # zeroed out afterwards, or it won't matter if you also mask out losses, etc.).
        if attention_mask is not None:
            # attention_mask: [b, n] with True = valid token, False = invalid/padded
            # Create 4D attention mask [batch_size, num_heads, seq_len, seq_len]
            # Add new dimensions: now shape is [b, 1, 1, n]
            attention_mask = attention_mask[:, None, None, :]
            # Expand across heads dimension: [b, h, 1, n].
            attention_mask = attention_mask.expand(-1, self.heads, -1, -1)
            # Mask out padding tokens with -inf
            dots = dots.masked_fill(~attention_mask, float('-inf'))

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

    Args:
        dim (int): Input dimension
        depth (int): Number of transformer layers
        heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head
        mlp_dim (int): Dimension of feed-forward network
        dropout (float, optional): Dropout probability. Defaults to 0.
        return_attention (bool, optional): Whether to return attention weights. Defaults to False.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., return_attention = False):
        super().__init__()
        self.return_attention = return_attention
        self.norm = nn.LayerNorm(dim)

        # Create stack of transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                         return_attention=return_attention),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, attention_mask=None):
        """Forward pass through transformer layers.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, dim]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            tuple:
                - torch.Tensor: Output tensor [batch_size, seq_len, dim]
                - list: Attention weights from each layer if return_attention=True
        """
        attention_weights = []
        for attn, ff in self.layers:
            attn_out, weights = attn(x, attention_mask)
            x = attn_out + x  # Attention with residual connection
            x = ff(x) + x    # Feed-forward with residual connection
            if self.return_attention:
                attention_weights.append(weights)

        return self.norm(x), attention_weights

class BertEncoder(nn.Module):
    """
    BERT-style transformer encoder with class token for sequence reconstruction.
    Uses a shared transformer encoder for all layers with pre-normalization and residual connections.

    Architecture:
        1. Token embeddings + Positional embeddings + CLS token
        2. Layer normalization and dropout
        3. N transformer layers with self-attention and feed-forward blocks
        4. Final layer normalization
        5. Output projection to vocabulary size

    Args:
        vocab_size (int): Size of vocabulary including special tokens
        hidden_size (int): Size of hidden layers (default: 768)
        num_layers (int): Number of transformer layers (default: 12)
        num_heads (int): Number of attention heads (default: 12)
        mlp_dim (int): Dimension of the MLP layer (default: 3072)
        pool (str): Pooling type, either 'cls' or 'mean' (default: 'cls')
        dropout (float): Dropout probability for attention and FFN (default: 0.1)
        emb_dropout (float): Dropout rate for embeddings (default: 0.1)
        max_seq_length (int): Maximum sequence length including CLS token (default: 512)
        return_attention (bool): Whether to return attention weights (default: False)

    Shape:
        - Input:
            - input_ids: (batch_size, seq_len)
            - attention_mask: (batch_size, seq_len) or None
        - Output:
            - logits: (batch_size, seq_len, vocab_size)
            - attention_weights: List of tensors of shape (batch_size, num_heads, seq_len, seq_len) if return_attention=True
    """
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12,
                 mlp_dim=3072, pool='cls', dropout=0.1, emb_dropout=0.1,
                 max_seq_length=512, return_attention=False):
        super().__init__()

        # Token and positional embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # Token embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length + 1, hidden_size))  # +1 for CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # Learnable CLS token
        self.dropout = nn.Dropout(emb_dropout)  # Embedding dropout

        # Configuration
        self.return_attention = return_attention
        self.pool = pool
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        # Main transformer encoder
        self.transformer = Transformer(
            dim=hidden_size,              # Hidden dimension size
            depth=num_layers,             # Number of transformer layers
            heads=num_heads,              # Number of attention heads
            dim_head=hidden_size // num_heads,  # Dimension of each attention head
            mlp_dim=mlp_dim,             # MLP dimension in feed-forward network
            dropout=dropout,              # Dropout rate
            return_attention=return_attention  # Whether to return attention weights
        )

        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size)

        # Initialize special tokens with smaller values for stability
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the BERT encoder.

        Process:
            1. Convert input IDs to embeddings
            2. Prepend CLS token to sequence
            3. Add positional embeddings
            4. Apply dropout
            5. Process through transformer layers
            6. Remove CLS token
            7. Project to vocabulary size

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask for padding.
                Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            tuple: Contains:
                - logits (torch.Tensor): Token prediction logits [batch_size, seq_len, vocab_size]
                - attention_weights (list, optional): List of attention weights if return_attention=True
        """
        b, n = input_ids.shape

        # Get embeddings from input IDs
        x = self.embedding(input_ids)  # [b, n, hidden_size]

        # Prepend CLS token to sequence
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, hidden_size]

        # Add positional embeddings and apply dropout
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Update attention mask to include CLS token
        if attention_mask is not None:
            cls_mask = torch.ones(b, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # Pass through transformer encoder
        x, attention_weights = self.transformer(x, attention_mask)

        # Remove CLS token from output
        x = x[:, 1:]  # [b, n, hidden_size]

        # Project to vocabulary size
        logits = self.output(x)  # [b, n, vocab_size]

        return logits, attention_weights if self.return_attention else None

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

def inspect_batch(batch, id2label, num_examples=1, logger=None):
    """Single-line batch inspection showing input and target sequences."""
    def log_msg(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    for i in range(min(num_examples, len(batch['input_ids']))):
        input_seq = ' '.join([id2label[id.item()]
                            for id in batch['input_ids'][i]
                            if id.item() != batch['pad_id']])
        label_seq = ' '.join([id2label[id.item()]
                            for id in batch['labels'][i]
                            if id.item() != batch['pad_id']])
        log_msg(f"Inspect: Batch[{i}] Input: {input_seq}")
        log_msg(f"Inspect: Batch[{i}] Label: {label_seq}")

def train_channel(model, optimizer, scheduler, train_loader, dev_loader, num_epochs, save_epoch_interval, result_dir, mtest_loader=None):
    """
    Train BERT model to minimize conditional entropy H(X|Y) by reconstructing masked sequences.

    Args:
        model (BertEncoder): The BERT model
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        train_loader (DataLoader): DataLoader for training data
        dev_loader (DataLoader): DataLoader for development/validation data
        num_epochs (int): Number of training epochs
        save_epoch_interval (int): Interval for saving model checkpoints
        result_dir (str): Directory to save results and checkpoints
        mtest_loader (DataLoader, optional): DataLoader for monitoring test set

    Returns:
        None: Model checkpoints and logs are saved to result_dir
    """
    # Setup result directory and logging
    overwrite_result_directory(result_dir, args.overwrite)
    logger = init_logger(os.path.join(result_dir, "report.log"))
    logger.info(args)

    # inspection
    id2label = {v: k for k, v in train_loader.dataset.label2id.items()}
    for batch_idx, batch in enumerate(train_loader):
        inspect_batch(batch, id2label, logger=logger)
        if batch_idx >= 0: break

    logger.info('Training new BERT channel model')

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    best_dev_loss = float('inf')
    best_dev_epoch = 0
    epoch = 0

    # Get pad_id from the training dataset
    pad_id = train_loader.dataset.pad_id


    while epoch < num_epochs:
        # Training phase
        model.train()
        train_loss = 0.0
        accuracies = []
        # Track variables for conditional entropy rate calculation
        total_log_prob_sum = 0.0    # Sum of negative log probabilities
        total_valid_tokens = 0      # Count of all valid (non-pad) tokens
        info_table = []

        for batch in tqdm.tqdm(train_loader, ascii=True, ncols=50):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            logits, _ = model(input_ids, attention_mask)

            # Calculate loss for all non-padding tokens
            # Reshape logits to (batch_size * seq_len, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            # Compute loss only on non-padding tokens
            valid_positions = (labels != pad_id)  # Use pad_id from dataset
            loss = F.cross_entropy(
                logits[valid_positions],
                labels[valid_positions],
                reduction='mean'
            ) / np.log(2)  # Convert from natural log to log base 2 (convert nat to bit)

            # Calculate conditional entropy statistics
            num_valid = valid_positions.sum().item()
            total_log_prob_sum += loss.item() * num_valid  # Convert mean loss back to sum
            total_valid_tokens += num_valid

            # Calculate accuracy on non-padding tokens
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions[valid_positions] == labels[valid_positions]).float().mean()
            accuracies.append(accuracy.item())

            # Backward pass
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
            optimizer.step()

            train_loss += loss.item()

            # Check for NaN
            if torch.isnan(loss):
                raise ValueError("NaN detected in training loss")

        # Calculate average training metrics
        train_loss /= len(train_loader) # Divided by the number of batches
        train_accuracy = np.mean(accuracies)
        # H(X|Y) = -1/N * Σ log P(X|Y) where N is total tokens
        train_cond_entropy_rate = total_log_prob_sum / total_valid_tokens
        train_avg_received_per_sent = 2**train_cond_entropy_rate
        train_avg_received_deviation = train_avg_received_per_sent - 1.0

        info_table.append([
            epoch,
            "train_set",
            f"{train_loss:.6e}",
            f"{train_accuracy:.6e}",
            f"{train_cond_entropy_rate:.6e}",  # bits
            f"1{train_avg_received_deviation:+.6e}"  # Format as 1±small_value
        ])

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits/train', train_cond_entropy_rate, epoch)
        writer.add_scalar('AvgReceivedPerSentDeviation/train', train_avg_received_deviation, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        accuracies = []
        # Reset entropy tracking for dev set
        total_log_prob_sum = 0.0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(dev_loader, ascii=True, ncols=50):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                logits, _ = model(input_ids, attention_mask)

                # Use same non-padding token loss calculation as training
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                valid_positions = (labels != pad_id)
                loss = F.cross_entropy(
                    logits[valid_positions],
                    labels[valid_positions],
                    reduction='mean'
                ) / np.log(2)  # Convert from natural log to log base 2 (convert nat to bit)

                # Calculate conditional entropy statistics for dev set
                num_valid = valid_positions.sum().item()
                total_log_prob_sum += loss.item() * num_valid
                total_valid_tokens += num_valid

                predictions = logits.argmax(dim=-1)
                accuracy = (predictions[valid_positions] == labels[valid_positions]).float().mean()
                accuracies.append(accuracy.item())

                eval_loss += loss.item()

        # Calculate dev set metrics
        eval_loss /= len(dev_loader)
        eval_accuracy = np.mean(accuracies)
        dev_cond_entropy_rate = total_log_prob_sum / total_valid_tokens
        dev_avg_received_per_sent = 2**dev_cond_entropy_rate
        dev_avg_received_deviation = dev_avg_received_per_sent - 1.0

        info_table.append([
            epoch,
            "dev_set",
            f"{eval_loss:.6e}",
            f"{eval_accuracy:.6e}",
            f"{dev_cond_entropy_rate:.6e}",
            f"1{dev_avg_received_deviation:+.6e}"  # Format as 1±small_value
        ])

        # Log dev metrics
        writer.add_scalar('Loss/dev', eval_loss, epoch)
        writer.add_scalar('Accuracy/dev', eval_accuracy, epoch)
        writer.add_scalar('ConditionalEntropyRate_bits/dev', dev_cond_entropy_rate, epoch)
        writer.add_scalar('AvgReceivedPerSentDeviation/dev', dev_avg_received_deviation, epoch)


        # Monitor test set if provided
        if mtest_loader:
            model.eval()
            mtest_loss = 0.0
            accuracies = []
            # Reset entropy tracking for test set
            total_log_prob_sum = 0.0
            total_valid_tokens = 0

            with torch.no_grad():
                for batch in tqdm.tqdm(mtest_loader, ascii=True, ncols=50):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    logits, _ = model(input_ids, attention_mask)

                    # Use same non-padding token loss calculation
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)

                    valid_positions = (labels != pad_id)
                    loss = F.cross_entropy(
                        logits[valid_positions],
                        labels[valid_positions],
                        reduction='mean'
                    ) / np.log(2)  # Convert from natural log to log base 2 (convert nat to bit)

                    # Calculate conditional entropy statistics for test set
                    num_valid = valid_positions.sum().item()
                    total_log_prob_sum += loss.item() * num_valid
                    total_valid_tokens += num_valid

                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions[valid_positions] == labels[valid_positions]).float().mean()
                    accuracies.append(accuracy.item())

                    mtest_loss += loss.item()

            # Calculate test set metrics
            mtest_loss /= len(mtest_loader)
            mtest_accuracy = np.mean(accuracies)
            test_cond_entropy_rate = total_log_prob_sum / total_valid_tokens
            test_avg_received_per_sent = 2**test_cond_entropy_rate
            test_avg_received_deviation = test_avg_received_per_sent - 1.0

            info_table.append([
                epoch,
                "test_set",
                f"{mtest_loss:.6e}",
                f"{mtest_accuracy:.6e}",
                f"{test_cond_entropy_rate:.6e}",
                f"1{test_avg_received_deviation:+.6e}"  # Format as 1±small_value
            ])

            # Log test metrics
            writer.add_scalar('Loss/test', mtest_loss, epoch)
            writer.add_scalar('Accuracy/test', mtest_accuracy, epoch)
            writer.add_scalar('ConditionalEntropyRate_bits/test', test_cond_entropy_rate, epoch)
            writer.add_scalar('AvgReceivedPerSentDeviation/test', test_avg_received_deviation, epoch)


        # Log metrics table with updated headers
        logger.info("\n" + tabulate.tabulate(
            info_table,
            headers=['epoch', 'dataset', 'loss', 'acc', 'cond_entropy_bits', 'avg_received_per_sent'],
            floatfmt='.6e',
            tablefmt='rst'
        ))

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")

        # Save periodic checkpoints
        if epoch % save_epoch_interval == 0:
            checkpoint_path = os.path.join(result_dir, f"model_e{epoch}.ckpt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Model saved: {checkpoint_path}")

        # Save best model on dev set
        if eval_loss < best_dev_loss:
            best_dev_loss = eval_loss
            best_dev_epoch = epoch
            logger.info(f"New best dev loss {best_dev_loss:.3f} at epoch {best_dev_epoch}")
            best_model_path = os.path.join(result_dir, "model_best_dev.ckpt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, best_model_path)
            logger.info(f"Best model saved: {best_model_path}")

        # Save latest model
        latest_model_path = os.path.join(result_dir, "model.ckpt")
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

"""
Main function to run the training pipeline for conditional entropy calculation.
Uses BERT/ViT base configuration parameters by default.
"""
parser = argparse.ArgumentParser(
    description='Train BERT model to minimize conditional entropy H(X|Y) for phoneme/syllable reconstruction'
)

# Data arguments
parser.add_argument('--train_corpus', type=str,
                    default='exp/data/sample_tv07_phoneme.txt',
                    help='Path to training corpus')
parser.add_argument('--dev_corpus', type=str,
                    default='exp/data/sample_tv07_phoneme.txt',
                    help='Path to development corpus')
parser.add_argument('--mtest_corpus', type=str,
                    default=None,
                    help='Path to monitoring test corpus (optional)')
parser.add_argument('--label2id_file', type=str,
                    default='conf/dict/chinese_all_label2id.txt',
                    help='Path to label2id mapping file')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Training batch size')
parser.add_argument('--max_seq_length', type=int, default=512,
                    help='Maximum sequence length')
parser.add_argument('--syllable_fl', action='store_true',
                    help='If specified, compute syllable-level forward length; otherwise compute phoneme-level')
parser.add_argument('--connector', type=str,
                   default='_',
                   help='Character used to connect phonemes in syllables')

# Model arguments (BERT/ViT Base configuration)
parser.add_argument('--hidden_size', type=int, default=384,
                    help='Hidden size of transformer (d_model)')
parser.add_argument('--num_layers', type=int, default=6,
                    help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=6,
                    help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout probability')

# Optimizer arguments
parser.add_argument('--base_lr', type=float, default=1e-4,
                    help='Base learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay for AdamW')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of warmup epochs')
parser.add_argument('--total_epochs', type=int, default=100,
                    help='Total epochs for scheduler cycle')
parser.add_argument('--constant_lr', action='store_true',
                    help='Use constant learning rate instead of batch size scaling')

# Training arguments
parser.add_argument('--seed', type=int, default=2025,
                    help='Random seed for reproducibility')
parser.add_argument('--num_epochs', type=int, default=3, # 100,
                    help='Number of training epochs')
parser.add_argument('--save_epoch_interval', type=int, default=10,
                    help='Save model checkpoint every N epochs')
parser.add_argument('--result_dir', type=str,
                    default='exp/fl_ce_chinese/test/sample_tv07_phoneme',
                    help='Directory to save results and checkpoints')

# Device arguments
parser.add_argument('--gpu', type=str, default='auto',
                    help='GPU device to use (auto for least memory usage)')

# Other arguments
parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing result directory')
parser.add_argument('--exit', action='store_true',
                    help='Exit after training without prompting to continue')

# Parse arguments
args = parser.parse_args()

# Setup
set_seed(args.seed)
device = set_device(args.gpu)
# Load vocabulary
label2id = {}
with open(args.label2id_file, 'r') as f:
    for line in f:
        token, idx = line.strip().split(': ')
        label2id [token.strip()] = int(idx)

# Create model
model = BertEncoder(
    vocab_size=len(label2id),
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout,
    max_seq_length=args.max_seq_length
).to(device)

# Create datasets
train_dataset = PhonemeDataset(
    args.train_corpus,
    label2id,
    max_seq_length=args.max_seq_length,
    syllable_fl=args.syllable_fl,
    connector=args.connector
)
dev_dataset = PhonemeDataset(
    args.dev_corpus,
    label2id,
    max_seq_length=args.max_seq_length,
    syllable_fl=args.syllable_fl,
    connector=args.connector
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# Create test dataset and loader if specified
mtest_loader = None
if args.mtest_corpus:
    mtest_dataset = PhonemeDataset(
        args.mtest_corpus,
        label2id,
        max_seq_length=args.max_seq_length,
        syllable_fl=args.syllable_fl,
        connector=args.connector
    )
    mtest_loader = DataLoader(mtest_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# Setup training
optimizer, scheduler = setup_optimizer_and_scheduler(
    model=model,
    base_lr=args.base_lr,
    batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_epochs=args.warmup_epochs,
    total_epochs=args.total_epochs,
    constant_base_lr=args.constant_lr
)

# Train
train_channel(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    dev_loader=dev_loader,
    num_epochs=args.num_epochs,
    save_epoch_interval=args.save_epoch_interval,
    result_dir=args.result_dir,
    mtest_loader=mtest_loader
)
