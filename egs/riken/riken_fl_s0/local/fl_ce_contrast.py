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
import csv

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

def process_contrast_groups(groups_str):
    """
    Process contrast group string into mappings for batch operations.

    Args:
        groups_str (str or None): Groups separated by '%', phonemes space-separated
                                 Example: "a b c%d e%f g" creates three groups

    Returns:
        tuple: Contains:
            - dict: Maps each phoneme to its contrast group set
            - dict: Maps each phoneme to its mask token

    Examples:
        >>> process_contrast_groups("a b c")  # Single group
        ({'a': {'a','b','c'}, 'b': {'a','b','c'}, 'c': {'a','b','c'}},
         {'a': '<mask>', 'b': '<mask>', 'c': '<mask>'})

        >>> process_contrast_groups("a b%c d")  # Two groups
        ({'a': {'a','b'}, 'b': {'a','b'}, 'c': {'c','d'}, 'd': {'c','d'}},
         {'a': '<mask>', 'b': '<mask>', 'c': '<mask1>', 'd': '<mask1>'})
    """
    if not groups_str:
        return {}, {}

    contrast_map = {}   # Phoneme to group mapping
    mask_map = {}       # Phoneme to mask token mapping

    groups = groups_str.strip().split('%')
    for group_idx, group in enumerate(groups):
        phonemes = sorted(group.strip().split())  # Sort phonemes for deterministic order

        # First group uses <mask>, additional groups use <mask1>, <mask2>, etc.
        mask_token = '<mask>' if group_idx == 0 else f'<mask{group_idx}>'

        # Map each phoneme to its group and mask token
        for p in phonemes:
            contrast_map[p] = phonemes  # Store sorted list instead of set
            mask_map[p] = mask_token

    return contrast_map, mask_map

def verify_mask_tokens(label2id, contrast_groups, merge_type):
    """
    Verify that required mask tokens exist in vocabulary for contrast merging.

    Args:
        label2id (dict): Dictionary mapping tokens to IDs
        contrast_groups (str): Groups string like "a b c%d e"
        merge_type (str): Either 'uniform' or 'mask'

    Returns:
        tuple: (bool, set) - (verification passed, missing mask tokens)

    Raises:
        ValueError: If merge_type='mask' and required mask tokens are missing

    Example:
        >>> verify_mask_tokens(label2id, "a b c%d e", "mask")
        # Checks for <mask> and <mask1>
        >>> verify_mask_tokens(label2id, "a b%c d%e f", "mask")
        # Checks for <mask>, <mask1>, and <mask2>
    """
    if merge_type != 'mask':
        return True, set()

    required_masks = {'<mask>'}  # Always need basic mask

    if contrast_groups:
        num_groups = contrast_groups.count('%')
        if num_groups > 0:
            required_masks.update(f'<mask{i}>' for i in range(1, num_groups + 1))

    missing_masks = required_masks - set(label2id.keys())
    if missing_masks:
        return False, missing_masks

    return True, set()

def save_utterance_metrics(batch_metrics, output_dir, dataset_name):
    """
    Save utterance-level metrics to JSON file.

    Args:
        batch_metrics (list): List of dictionaries containing utterance metrics
        output_dir (str): Directory to save JSON files
        dataset_name (str): Name of dataset (train/dev/test)
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_utterance_fl.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_metrics, f, ensure_ascii=False, indent=2)

def compute_utterance_metrics(model, data_loader, device, pad_id, contrast_groups, merge_type, epoch):
    """
    Compute utterance-level metrics including token probabilities and functional load.

    This function processes each utterance to calculate:
    1. Original and merged token sequences
    2. Individual token probabilities P(X|Y)
    3. Mean negative log probability in bits (both as loss and conditional entropy)
    4. Token-level prediction accuracy
    5. Predicted token sequence

    Args:
        model: Neural model for token prediction
        data_loader: DataLoader containing batched utterances
        device: Device (CPU/GPU) for computations
        pad_id: ID of padding token to mask out
        contrast_groups: String describing phoneme contrast groups
        merge_type: Type of contrast merging ('uniform'/'mask')
        epoch: Current training epoch

    Returns:
        list of dict: List of dictionaries containing metrics for each utterance:
            - original_utterance: Original token sequence
            - merged_utterance: Sequence after contrast merging
            - predicted_utterance: Model's predicted token sequence
            - token_probabilities: List of P(X|Y) for each token
            - cond_entropy_bits: Mean negative log probability in bits
            - loss: Same as cond_entropy_bits (per-token average)
            - accuracy: Ratio of correctly predicted tokens
            - contrast_groups: Contrast group configuration
            - merge_type: Type of contrast merging
            - epoch: Training epoch
            - utterance_length: Number of tokens
    Note:
        Both loss and conditional entropy are computed as mean(-log2(P(X|Y))),
        representing the average number of bits needed per token.
        This matches standard practice where loss is reported as a per-token average.

    """
    model.eval()  # Set model to evaluation mode
    utterance_metrics = []

    # Get id2label mapping for converting predictions back to tokens
    id2label = {v: k for k, v in data_loader.dataset.label2id.items()}

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm.tqdm(data_loader, ascii=True, ncols=50):
            # Move batch data to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_tokens = batch['original_tokens']
            merged_tokens = batch['merged_tokens']

            # Get model predictions
            logits, _ = model(input_ids, attention_mask)

            # Process each utterance in batch
            for i in range(input_ids.size(0)):
                # Mask out padding tokens
                valid_positions = labels[i] != pad_id
                utterance_logits = logits[i][valid_positions]
                utterance_labels = labels[i][valid_positions]

                # Convert logits to probabilities
                probs = F.softmax(utterance_logits, dim=-1)

                # Get predictions and convert to tokens
                predictions = utterance_logits.argmax(dim=-1)
                predicted_tokens = [id2label[pred.item()] for pred in predictions]

                # Extract probability for each correct token
                token_probs = []
                for j, label in enumerate(utterance_labels):
                    token_probs.append(probs[j, label].item())

                # Calculate negative log probabilities in bits
                token_log_probs = [-math.log2(p) for p in token_probs]

                # Calculate per-token average (used for both loss and entropy)
                mean_neg_log_prob = sum(token_log_probs) / len(token_log_probs)

                # Calculate prediction accuracy
                accuracy = (predictions == utterance_labels).float().mean().item()

                # Store metrics for this utterance
                utterance_metrics.append({
                    'original_utterance': ' '.join(original_tokens[i]),
                    'merged_utterance': ' '.join(merged_tokens[i]),
                    'predicted_utterance': ' '.join(predicted_tokens),  # Add predicted utterance
                    'token_probabilities': token_probs,
                    'cond_entropy_bits': mean_neg_log_prob,  # Average bits per token
                    'loss': mean_neg_log_prob,  # Same as cond_entropy (per-token average)
                    'accuracy': accuracy,
                    'contrast_groups': contrast_groups,
                    'merge_type': merge_type,
                    'epoch': epoch,
                    'utterance_length': len(original_tokens[i])
                })

    return utterance_metrics

class PhonemeDataset(Dataset):
    """
    Dataset for syllable sequences with phoneme or syllable-level forward length computation.

    Input is always in syllable format with phonemes connected by connector character.
    The syllable_fl parameter controls whether to compute forward length at syllable
    or phoneme level.

    Args:
        data_path (str): Path to corpus file containing syllable sequences
        label2id (dict): Dictionary mapping tokens to integer IDs
        max_seq_length (int, optional): Maximum sequence length. Defaults to 512.
        syllable_fl (bool, optional): If True, compute forward length using syllables.
                                    If False, split syllables into phonemes. Defaults to False.
        connector (str, optional): Character connecting phonemes in syllables. Defaults to '_'.

    Example input format (always syllables):
        c_ai f_u g_u sh_iii h_ui
        c_ai f_u x_in g_uan n_ian

    When syllable_fl=True:
        Keeps syllables intact: ['c_ai', 'f_u', 'g_u', 'sh_iii', 'h_ui']

    When syllable_fl=False:
        Splits syllables into phonemes: ['c', 'ai', 'f', 'u', 'g', 'u', 'sh', 'iii', 'h', 'ui']
    """
    def __init__(self, data_path, label2id, max_seq_length=512, syllable_fl=False, connector='_'):
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.connector = connector
        self.syllable_fl = syllable_fl
        self.pad_id = label2id['<pad>']

        # Load sequences preserving original tokens for contrast merging
        self.sequences = []       # Store token ID sequences
        self.original_tokens = [] # Store original tokens for merging

        with open(data_path, 'r') as f:
            for line_idx, line in enumerate(f, 1):
                original_line = line.strip()
                syllables = original_line.split()

                if syllable_fl:
                    # Keep syllables intact for syllable-level FL
                    tokens = syllables
                else:
                    # Split syllables into phonemes for phoneme-level FL
                    tokens = [phoneme
                            for syllable in syllables
                            for phoneme in syllable.split(self.connector)]

                # Check for unknown tokens
                unknowns = [t for t in tokens if t not in label2id]
                if unknowns:
                    print(f"Warning: Unknown tokens in {data_path} at line {line_idx}: {unknowns}")
                    print(f"Utterance: {original_line}")

                if len(tokens) > 0:
                    self.original_tokens.append(tokens)
                    ids = [label2id.get(token, label2id['<unk>'])
                          for token in tokens]
                    self.sequences.append(ids)

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single sequence and its original tokens for batch processing.

        Args:
            idx (int): Index of sequence to retrieve

        Returns:
            dict: Contains:
                - input_ids: Tensor of token ids
                - original_tokens: Original tokens for contrast merging
                - pad_id: The padding token ID
        """
        # Get sequence and truncate if needed
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_length:
            print(f"Warning: Sequence at index {idx} exceeds max_seq_length ({len(seq)} > {self.max_seq_length}). Truncating.")
            seq = seq[:self.max_seq_length]

        return {
            'input_ids': torch.tensor(seq),
            'original_tokens': self.original_tokens[idx],
            'pad_id': self.pad_id,
            'syllable_fl': self.syllable_fl
        }

def train_collate_fn(batch, contrast_map=None, mask_map=None, merge_type=None,
                     connector='_', label2id=None):
    """
    Collate function that handles padding and contrast merging during batch creation.

    This function:
    1. Performs dynamic contrast merging if contrast groups are specified
    2. Pads sequences to the maximum length in the current batch
    3. Creates attention masks for valid positions

    This enables more efficient processing by only padding to the necessary length
    for each batch, rather than padding all sequences to the global max_seq_length.

    Args:
        batch: List of dictionaries from PhonemeDataset, where each dictionary contains:
            - input_ids (torch.Tensor): Unpadded token IDs [seq_len]
            - original_tokens (list): Original tokens for contrast merging
            - pad_id (int): The padding token ID
        contrast_map (dict, optional): Maps each phoneme to its contrast group set
        mask_map (dict, optional): Maps each phoneme to its group-specific mask token
        merge_type (str, optional): Either 'uniform' or 'mask' or None
        connector (str): Character used to connect phonemes in syllables
        label2id (dict): Dictionary mapping tokens to integer IDs

    Returns:
        dict: A dictionary containing batched and padded tensors:
            - input_ids (torch.Tensor): Padded and possibly merged input tensors [batch_size, max_batch_len]
            - labels (torch.Tensor): Padded original (unmerged) tensors [batch_size, max_batch_len]
            - attention_mask (torch.Tensor): Binary mask for valid positions [batch_size, max_batch_len]
                                          (1 for tokens, 0 for padding)

    Example outputs:
        Standard padding (no contrast merging):
            Input sequences: ["a b c", "d e"]
            Returns:
                input_ids: [[a, b, c, pad], [d, e, pad, pad]]
                labels:    [[a, b, c, pad], [d, e, pad, pad]]
                attention_mask: [[1, 1, 1, 0], [1, 1, 0, 0]]

        With contrast merging (--contrast_groups "a b c%d e" --merge_type mask):
            Input sequences: ["a b c", "d e"]
            Returns:
                input_ids: [[<mask>, <mask>, <mask>, pad], [<mask1>, <mask1>, pad, pad]]
                labels:    [[a, b, c, pad], [d, e, pad, pad]]
                attention_mask: [[1, 1, 1, 0], [1, 1, 0, 0]]

        With syllable contrast merging (--contrast_groups "sh ch%p b" --merge_type uniform):
            Input: ["sh_a ch_i", "p_u b_e"]
            Possible returns:
                input_ids: [[ch_a sh_i pad pad], [b_u p_e pad pad]]
                labels:    [[sh_a ch_i pad pad], [p_u b_e pad pad]]
                attention_mask: [[1, 1, 0, 0], [1, 1, 0, 0]]

    Note:
        Sequences longer than max_seq_length are already truncated in PhonemeDataset.__getitem__,
        so this function only needs to handle padding up to the longest sequence in the batch.
    """
    # Process each sequence in the batch
    processed_seqs = []
    original_seqs = []
    processed_tokens_list = []  # Store processed tokens for each item in batch

    for item in batch:
        original_tokens = item['original_tokens']
        original_seqs.append(item['input_ids'])
        syllable_fl = item['syllable_fl']  # Get syllable_fl flag

        if contrast_map and mask_map and merge_type:
            # Apply contrast merging to original tokens
            processed_tokens = []
            for token in original_tokens:
                if connector in token:  # Syllable processing
                    phonemes = token.split(connector)
                    processed_phonemes = []
                    for p in phonemes:
                        if p in contrast_map:
                            if merge_type == 'uniform':
                                p = random.choice(contrast_map[p])
                            else:  # mask
                                p = mask_map[p]
                        processed_phonemes.append(p)
                    if syllable_fl:
                        # Keep as syllable if syllable_fl is True
                        processed_tokens.append(connector.join(processed_phonemes))
                    else:
                        # Split into phonemes if syllable_fl is False
                        processed_tokens.extend(processed_phonemes)
                else:  # Single phoneme processing
                    if token in contrast_map:
                        if merge_type == 'uniform':
                            token = random.choice(contrast_map[token])
                        else:  # mask
                            token = mask_map[token]
                    processed_tokens.append(token)
            processed_tokens_list.append(processed_tokens)  # Store processed tokens

            if syllable_fl:
                # For syllable FL, process tokens as complete syllables
                seq_ids = []
                for token in processed_tokens:
                    token_id = label2id.get(token, label2id['<unk>'])
                    if token_id == label2id['<unk>']:
                        print(f"Warning: Unknown token '{token}' converted to <unk>")
                    seq_ids.append(token_id)
            else:
                # For phoneme FL, process tokens as phonemes
                seq_ids = []
                for token in processed_tokens:
                    if connector in token:
                        # Process syllable
                        phonemes = token.split(connector)
                        for p in phonemes:
                            p_id = label2id.get(p, label2id['<unk>'])
                            if p_id == label2id['<unk>']:
                                print(f"Warning: Unknown phoneme '{p}' in syllable '{token}' converted to <unk>")
                            seq_ids.append(p_id)
                    else:
                        # Process single phoneme
                        token_id = label2id.get(token, label2id['<unk>'])
                        if token_id == label2id['<unk>']:
                            print(f"Warning: Unknown token '{token}' converted to <unk>")
                        seq_ids.append(token_id)

            processed_seqs.append(torch.tensor(seq_ids))
        else:
            # No contrast merging - use original sequence
            processed_seqs.append(item['input_ids'])
            processed_tokens_list.append(original_tokens)  # Store processed tokens

    # Find maximum length in this batch for efficient padding
    pad_id = batch[0]['pad_id']
    max_len = max(len(seq) for seq in processed_seqs)

    # Pad sequences to max length in batch
    # Add pad_id elements to the right of sequences
    input_ids = [F.pad(seq, (0, max_len - len(seq)), value=pad_id)
                for seq in processed_seqs]
    labels = [F.pad(seq, (0, max_len - len(seq)), value=pad_id)
              for seq in original_seqs]

    # Stack tensors and create attention mask
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = (input_ids != pad_id)  # 1 for tokens, 0 for padding

    return {
        'input_ids': input_ids,      # Padded and possibly merged inputs
        'labels': labels,            # Padded original (unmerged) labels
        'pad_id': pad_id,
        'attention_mask': attention_mask,  # Mask for valid positions
        'original_tokens': [item['original_tokens'] for item in batch],
        'merged_tokens': processed_tokens_list  # Store the processed/merged tokens
    }

def eval_collate_fn(batch, contrast_map=None, mask_map=None, merge_type=None,
                    connector='_', label2id=None, seed=None):
    """
    Evaluation collate with deterministic contrast merging.
    Uses fixed seed for reproducible evaluation.
    """
    # Create a separate random generator for evaluation
    eval_rng = random.Random(seed) if seed is not None else random.Random()

    # Similar to train_collate_fn but with fixed randomization
    processed_seqs = []
    original_seqs = []
    processed_tokens_list = []  # Store processed tokens for each item in batch

    for item in batch:
        original_tokens = item['original_tokens']
        original_seqs.append(item['input_ids'])
        syllable_fl = item['syllable_fl']  # Get syllable_fl flag

        if contrast_map and mask_map and merge_type:
            processed_tokens = []
            for token in original_tokens:
                if connector in token:  # Syllable processing
                    phonemes = token.split(connector)
                    processed_phonemes = []
                    for p in phonemes:
                        if p in contrast_map:
                            if merge_type == 'uniform':
                                choices = sorted(contrast_map[p])
                                p = eval_rng.choice(choices)  # Use seeded RNG
                            else:  # mask
                                p = mask_map[p]
                        processed_phonemes.append(p)
                    if syllable_fl:
                        processed_tokens.append(connector.join(processed_phonemes))
                    else:
                        processed_tokens.extend(processed_phonemes)
                else:  # Single phoneme processing
                    if token in contrast_map:
                        if merge_type == 'uniform':
                            choices = sorted(contrast_map[token])
                            token = eval_rng.choice(choices)  # Use seeded RNG
                        else:
                            token = mask_map[token]
                    processed_tokens.append(token)
            processed_tokens_list.append(processed_tokens)  # Store processed tokens

            if syllable_fl:
                # For syllable FL, process tokens as complete syllables
                seq_ids = []
                for token in processed_tokens:
                    token_id = label2id.get(token, label2id['<unk>'])
                    if token_id == label2id['<unk>']:
                        print(f"Warning: Unknown token '{token}' converted to <unk>")
                    seq_ids.append(token_id)
            else:
                # For phoneme FL, process tokens as phonemes
                seq_ids = []
                for token in processed_tokens:
                    if connector in token:
                        # Process syllable
                        phonemes = token.split(connector)
                        for p in phonemes:
                            p_id = label2id.get(p, label2id['<unk>'])
                            if p_id == label2id['<unk>']:
                                print(f"Warning: Unknown phoneme '{p}' in syllable '{token}' converted to <unk>")
                            seq_ids.append(p_id)
                    else:
                        # Process single phoneme
                        token_id = label2id.get(token, label2id['<unk>'])
                        if token_id == label2id['<unk>']:
                            print(f"Warning: Unknown token '{token}' converted to <unk>")
                        seq_ids.append(token_id)

            processed_seqs.append(torch.tensor(seq_ids))
        else:
            processed_seqs.append(item['input_ids'])
            processed_tokens_list.append(original_tokens)  # Use original tokens when no merging


    # Regular padding and masking operations
    pad_id = batch[0]['pad_id']
    max_len = max(len(seq) for seq in processed_seqs)

    input_ids = [F.pad(seq, (0, max_len - len(seq)), value=pad_id)
                for seq in processed_seqs]
    labels = [F.pad(seq, (0, max_len - len(seq)), value=pad_id)
              for seq in original_seqs]

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = (input_ids != pad_id)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'pad_id': pad_id,
        'attention_mask': attention_mask,
        'original_tokens': [item['original_tokens'] for item in batch],
        'merged_tokens': processed_tokens_list  # Store the processed/merged tokens
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

def train_channel(model, optimizer, scheduler, train_loader, dev_loader, num_epochs,
                 save_epoch_interval, result_dir, mtest_loader=None, contrast_groups=None,
                 merge_type=None, pretrained_model=None, utterance_fl_json=False):
    """
    Train BERT model to minimize conditional entropy H(X|Y) by reconstructing masked sequences.
    If num_epochs=0, only runs evaluation on pretrained model without training.

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
        contrast_groups (str, optional): Configuration string for contrast groups
        merge_type (str, optional): Type of contrast merging used ('uniform' or 'mask')
        pretrained_model (str, optional): path to pretrained model (default None)
            When specified, would eval without training in epoch 0
        utterance_fl_json (bool, opitional): output the utterance fl and probalities (default: false)
    """
    # Setup result directory and logging
    overwrite_result_directory(result_dir, args.overwrite)
    logger = init_logger(os.path.join(result_dir, "report.log"))

    # Create stats directory and CSV files
    stats_dir = os.path.join(result_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    train_csv = os.path.join(stats_dir, 'train.csv')
    dev_csv = os.path.join(stats_dir, 'dev.csv')
    test_csv = os.path.join(stats_dir, 'test.csv')

    # Create CSV headers if they don't exist
    for csv_file in [train_csv, dev_csv, test_csv]:
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                csv_writer = csv.writer(f, lineterminator='\n')  # Explicitly set Unix line endings
                csv_writer.writerow(['epoch', 'dataset', 'loss', 'accuracy', 'cond_entropy_bits',
                               'avg_received_per_sent', 'contrast_groups', 'merge_type'])

    # inspection
    id2label = {v: k for k, v in train_loader.dataset.label2id.items()}
    for batch_idx, batch in enumerate(train_loader):
        inspect_batch(batch, id2label, logger=logger)
        if batch_idx >= 0: break

    logger.info(args)
    logger.info('Training new BERT channel model')

    if pretrained_model:
        checkpoint = torch.load(pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded pretrained model from {args.pretrained_model}")

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard'))

    best_dev_loss = float('inf')
    best_dev_epoch = 0
    epoch = 0

    # Get pad_id from the training dataset
    pad_id = train_loader.dataset.pad_id

    # Opition of saving the utterance fl and probabilities
    if utterance_fl_json: num_epochs = 1
    if utterance_fl_json and (not pretrained_model):
        raise ValueError("utterance_fl_json requires a pretrained model")

    if utterance_fl_json:
        prob_dir = os.path.join(result_dir, 'utterance_prob_fl')
        os.makedirs(prob_dir, exist_ok=True)

        # Compute and save metrics for each dataset
        for loader, name in [(train_loader, 'train'),
                           (dev_loader, 'dev'),
                           (mtest_loader, 'test')]:
            if loader is not None:
                metrics = compute_utterance_metrics(
                    model, loader, device, pad_id,
                    contrast_groups, merge_type, epoch=0
                )
                save_utterance_metrics(metrics, prob_dir, name)

        return

    # Evaluate pretrained model if starting from epoch 0
    if epoch == 0 and pretrained_model:
        epoch += 1
        model.eval()
        info_table = []

        with torch.no_grad():
            # Log initial metrics for all datasets
            for data_loader, csv_file, set_name in [
                (train_loader, train_csv, 'train_set'),
                (dev_loader, dev_csv, 'dev_set'),
                (mtest_loader, test_csv, 'test_set')
            ]:
                if data_loader is None:
                    continue

                # Evaluate and log metrics
                eval_loss = 0.0
                accuracies = []
                total_log_prob_sum = 0.0
                total_valid_tokens = 0

                for batch in tqdm.tqdm(data_loader, ascii=True, ncols=50):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    logits, _ = model(input_ids, attention_mask)
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)

                    valid_positions = (labels != pad_id)
                    loss = F.cross_entropy(
                        logits[valid_positions],
                        labels[valid_positions],
                        reduction='mean'
                    ) / np.log(2)

                    num_valid = valid_positions.sum().item()
                    total_log_prob_sum += loss.item() * num_valid
                    total_valid_tokens += num_valid

                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions[valid_positions] == labels[valid_positions]).float().mean()
                    accuracies.append(accuracy.item())

                    eval_loss += loss.item()

                # Calculate metrics
                eval_loss /= len(data_loader)
                eval_accuracy = np.mean(accuracies)
                cond_entropy_rate = total_log_prob_sum / total_valid_tokens
                avg_received_per_sent = 2**cond_entropy_rate
                avg_received_deviation = avg_received_per_sent - 1.0

                # Log to info table
                info_table.append([
                    0,                        # epoch 0
                    set_name,
                    f"{eval_loss:.6e}",
                    f"{eval_accuracy:.6e}",
                    f"{cond_entropy_rate:.6e}",
                    f"1{avg_received_deviation:+.6e}",
                ])

                # Log to CSV
                with open(csv_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f, lineterminator='\n')  # Explicitly set Unix line endings
                    csv_writer.writerow([
                        0,  # epoch 0
                        set_name,
                        f"{eval_loss:.6e}",
                        f"{eval_accuracy:.6e}",
                        f"{cond_entropy_rate:.6e}",
                        f"1{avg_received_deviation:+.6e}",
                        contrast_groups,
                        merge_type
                    ])

                # Log to tensorboard
                writer.add_scalar(f'Loss/{set_name.replace('_set', '')}', eval_loss, 0)
                writer.add_scalar(f'Accuracy/{set_name.replace('_set', '')}', eval_accuracy, 0)
                writer.add_scalar(f'ConditionalEntropyRate_bits/{set_name.replace('_set', '')}', cond_entropy_rate, 0)
                writer.add_scalar(f'AvgReceivedPerSentDeviation/{set_name.replace('_set', '')}', avg_received_deviation, 0)

        # Print the table for epoch 0
        logger.info("\n" + tabulate.tabulate(
            info_table,
            headers=['epoch', 'dataset', 'loss', 'acc', 'cond_entropy_bits', 'avg_received_per_sent'],
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

        # Add CSV logging for each dataset after metrics are calculated
        for metrics, csv_file in [
            ((train_loss, train_accuracy, train_cond_entropy_rate, train_avg_received_deviation), train_csv),
            ((eval_loss, eval_accuracy, dev_cond_entropy_rate, dev_avg_received_deviation), dev_csv),
            ((mtest_loss, mtest_accuracy, test_cond_entropy_rate, test_avg_received_deviation), test_csv) if mtest_loader else (None, None)
        ]:
            if metrics is not None:
                loss, acc, entropy, deviation = metrics
                with open(csv_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f, lineterminator='\n')  # Explicitly set Unix line endings
                    csv_writer.writerow([
                        epoch,
                        'train_set' if csv_file == train_csv else 'dev_set' if csv_file == dev_csv else 'test_set',
                        f"{loss:.6e}",
                        f"{acc:.6e}",
                        f"{entropy:.6e}",
                        f"1{deviation:+.6e}",
                        contrast_groups,
                        merge_type
                    ])

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
                    help='If specified, compute forward length using syllables instead of phonemes')
parser.add_argument('--connector', type=str,
                   default='_',
                   help='Character used to connect phonemes in syllables')
parser.add_argument('--utterance_fl_json', action='store_true',
                    help='Output utterance-level functional load metrics to JSON files')

# Contrast merge arguments
parser.add_argument('--contrast_groups', type=str, default=None,
                    help='Groups of contrasting phonemes separated by "%". '
                         'Phonemes within each group are space-separated. '
                         'Example: "a b c%d e" creates two groups: (a,b,c) and (d,e). '
                         'During training with uniform merge: any phoneme in group randomly '
                         'becomes any phoneme in same group. With mask merge: any phoneme '
                         'in group becomes <mask>. e.g., "a b c%d e"'
                         'first group uses <mask>, second uses <mask1>')
parser.add_argument('--merge_type', type=str, choices=['uniform', 'mask'], default='uniform',
                    help='Type of contrast merging: '
                         '"uniform" for random sampling within contrast group, '
                         '"mask" for replacing with progressive mask tokens')

# Model arguments (BERT/ViT Base configuration)
parser.add_argument('--hidden_size', type=int, default=384,
                    help='Hidden size of transformer (d_model)')
parser.add_argument('--num_layers', type=int, default=6,
                    help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=6,
                    help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout probability')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='Path to pretrained model checkpoint')

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

# Process contrast groups
contrast_map, mask_map = process_contrast_groups(args.contrast_groups)

# <mask> check when using mask merging
verification_passed, missing_masks = verify_mask_tokens(label2id, args.contrast_groups, args.merge_type)
if not verification_passed: raise ValueError(f"Missing required mask tokens in label2id: {missing_masks}")

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

# Create data loaders with appropriate collate functions
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=lambda batch: train_collate_fn(
        batch,
        contrast_map=contrast_map,
        mask_map=mask_map,
        merge_type=args.merge_type,
        connector=args.connector,
        label2id=label2id
    )
)

# Use eval_collate_fn with fixed seed for dev/test
dev_loader = DataLoader(
    dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=lambda batch: eval_collate_fn(
        batch,
        contrast_map=contrast_map,
        mask_map=mask_map,
        merge_type=args.merge_type,
        connector=args.connector,
        label2id=label2id,
        seed=args.seed  # Use fixed seed for evaluation
    )
)

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
    mtest_loader = DataLoader(
        mtest_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: eval_collate_fn(
            batch,
            contrast_map=contrast_map,
            mask_map=mask_map,
            merge_type=args.merge_type,
            connector=args.connector,
            label2id=label2id,
            seed=args.seed
        )
    )

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
    mtest_loader=mtest_loader,
    contrast_groups=args.contrast_groups,
    merge_type=args.merge_type,
    pretrained_model=args.pretrained_model,
    utterance_fl_json=args.utterance_fl_json
)
