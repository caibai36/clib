# Implemented by bin-wu at 02:52 on 10 April 2024 from the MIT's implementation
#    -> v1: Separate the training and the testing codes.
#           Add a logger.
#    -> v2: Change interations to epochs
#           Add a dataloader for training and development sets

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
parser = argparse.ArgumentParser(description=("Train or evaluate MIT CNN 72. (Reference: 'new_train_72.py' from https://marmosetbehavior.mit.edu/, supporting TensorFlow 2)"))

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
parser.add_argument("--test_pred1", type=str, default="exp/run/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos", help="Path to save the predicted label probabilities of the first stream for the test set")
parser.add_argument("--test_pred2", type=str, default="exp/run/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos", help="Path to save the predicted label probabilities of the second stream for the test set")

# Model arguments
parser.add_argument("--batch_size", type=int, default=25, help="Batch size for the dataloader")
parser.add_argument("--dropout_rate", type=float, default=0.4, help="Drop rate of dropout layer")
parser.add_argument("--eval_model", type=str, default="exp/run/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/model.ckpt", help="Model path for prediction or evaluation") # pred

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
parser.add_argument('--gpu', type=str, default='auto', # if 'auto', running three times in ipython will occupy three different GPUs.
                    help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least GPU memory")
parser.add_argument("--result", type=str, default="exp/run/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/", help="Result directory")
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
        model (TwoStreamCNNModel): The model to be trained.
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
            _, _, loss, accurs = model(batch_xs1, batch_xs2, batch_labels1, batch_labels2)
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
                _, _, loss, accurs = model(batch_xs1, batch_xs2, batch_labels1, batch_labels2)
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
        model (TwoStreamCNNModel): The trained model used for prediction.
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
        preds, preds2 = model(torch.from_numpy(inputs['x']).to(device),
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
    model = TwoStreamCNNModel(dropout_rate).to(device)
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
