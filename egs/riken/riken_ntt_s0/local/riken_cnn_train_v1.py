# Implemented by bin-wu at 17:29 on 18 April 2024

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

class OneStreamCNNModel(nn.Module):
    """
    A one-stream convolutional neural network model.

    Args:
    dropout_rate (float, optional): The dropout rate. Defaults to 0.5.
    num_classes (int): The number of categories for the classification
    """

    def __init__(self, num_classes, dropout_rate=0.5):
        super(OneStreamCNNModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Convolutional stream
        self.conv = nn.Sequential(
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
        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(1024, self.num_classes)

    def forward(self, x, y=None, mode='train'):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the one stream, of shape (batch_size, 257, 256).
            y (torch.Tensor, optional): Ground truth labels for output, of shape (batch_size,). Defaults to None.
            mode (str, optional): The mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            If y is provided:
                tuple: A tuple containing:
                    - probs (torch.Tensor): Probabilities for the output, of shape (batch_size, num_classes).
                    - loss (torch.Tensor): The computed loss.
                    - accuracy (list): A list of accuracies for each sample in the batch.
            If y is not provided:
                    - probs (torch.Tensor): Probabilities for the output, of shape (batch_size, num_classes).
        """
        # Pass input x through the convolutional stream
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        # Pass the features through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Obtain the logits for the output
        logits = self.fc2(x)

        # Compute the probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # If ground truth labels are provided, compute loss and accuracy
        if y is not None:
            y = y.type(torch.LongTensor).to(y.device)
            loss = F.cross_entropy(logits, y)
            classes = logits.argmax(dim=1)
            accuracy = (classes == y).float()
            return probs, loss, accuracy.tolist()

        # If ground truth labels are not provided, return only the probabilities
        return probs

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
    Function for training the network using the training and evaluation/development data loaders.

    Args:
        model (OneStreamCNNModel): The model to be trained.
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
            _, loss, accurs = model(batch_xs, batch_labels)
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
                _, loss, accurs = model(batch_xs, batch_labels)
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
                    _, loss, accurs = model(batch_xs, batch_labels)
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
    Function for predicting with the trained network for the testing set.

    Args:
        model (OneStreamCNNModel): The trained model used for prediction.
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
        preds = model(torch.from_numpy(inputs).to(device))
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
exp_dir = "exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs25lr0.0003evalinterval1avgpredwin5"
# Argument parser
parser = argparse.ArgumentParser(description=("Train or evaluate CNN."))
# Data arguments
parser.add_argument("--train_input", type=str, default=data_dir+"/train_input.npy", help="Path to the input of the training set")
parser.add_argument("--train_target", type=str, default=data_dir+"/train_target.npy", help="Path to the target labels of the training set")
parser.add_argument("--dev_input", type=str, default=data_dir+"/dev_input.npy", help="Path to the input of the development set")
parser.add_argument("--dev_target", type=str, default=data_dir+"/dev_target.npy", help="Path to the target labels of the development set")
parser.add_argument("--mtest_input", type=str, default=data_dir+"/test_input.npy", help="Path to the input of the test set for monitoring")
parser.add_argument("--mtest_target", type=str, default=data_dir+"/test_target.npy", help="Path to the target labels of the test set for monitoring")
parser.add_argument("--test_input", type=str, default=data_dir+"/test_input_Athos.npy", help="Path to the input of the testing set")
parser.add_argument("--test_pred", type=str, default=exp_dir+"/eval/test_pred_Athos.npy", help="Path to save the predicted label probabilities for the test set")
parser.add_argument("--batch_size", type=int, default=25, help="Batch size for the dataloader")
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id_marmoset.yaml", help="The YAML file that contains the label-to-labelID mapping.")

# Model arguments
parser.add_argument("--dropout_rate", type=float, default=0.4, help="Drop rate of dropout layer")

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
dropout_rate = args.dropout_rate
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

# Training or evaluation the model
model = OneStreamCNNModel(num_classes=len(label2id), dropout_rate=dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)

# Training stage
start_time = datetime.datetime.now()
train_loader = create_dataloader(train_input, train_target, batch_size, train=True)
dev_loader = create_dataloader(dev_input, dev_target, batch_size, train=False)
mtest_loader = create_dataloader(mtest_input, mtest_target, batch_size, train=False) if mtest_input and mtest_target else None # test set for monitoring
train(model, optimizer, train_loader, dev_loader, num_epochs, save_epoch_interval, lr, lr_decay_interval, result, mtest_loader)
duration = datetime.datetime.now() - start_time
print(f'Time taken to complete the training: {duration.seconds // 3600:02}:{(duration.seconds // 60) % 60:02}:{duration.seconds % 60:02}')

# # Prediction stage
# if not os.path.exists(os.path.dirname(test_pred)): os.makedirs(os.path.dirname(test_pred))
# logger = init_logger(os.path.join(os.path.dirname(test_pred), "report.log")) # logger for testing
# predictions = predict(model, test_input, model_path=eval_model, avg_pred_win=avg_pred_win, batch_size=batch_size)
# np.save(test_pred, predictions)
