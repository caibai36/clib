# Implemented by bin-wu at 16:21 on 9 April 2024 from the MIT's implementation
#    -> v1: Separate the training and the testing codes.
#           Add a logger.

import os
import sys
import datetime

import math
import random
import argparse

import GPUtil

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
parser.add_argument("--test_pred1", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos", help="Path to save the predicted label probabilities of the first stream for the test set")
parser.add_argument("--test_pred2", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos", help="Path to save the predicted label probabilities of the second stream for the test set")

# Model arguments
parser.add_argument("--batch_size", type=int, default=25, help="Batch size for the dataloader")
parser.add_argument("--dropout_rate", type=float, default=0.4, help="Drop rate of dropout layer")
# parser.add_argument("--eval_model", type=str, default="", help="Model path for prediction or evaluation") # train
parser.add_argument("--eval_model", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/model.ckpt", help="Model path for prediction or evaluation") # pred

# Optimizer arguments
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate of Adam optimizer")
parser.add_argument("--epsilon", type=float, default=0.001, help="Epsilon of Adam optimizer for numerical stability")

# Training arguments
parser.add_argument("--num_iter", type=int, default=2601, help="Number of iterations")
parser.add_argument("--eval_interval", type=int, default=200, help="Evaluate the development set every x iterations")

# Evaluation arguments
parser.add_argument("--avg_pred_win", type=int, default=5, help="Collect predicted probabilities by averaging across x consecutive predictions")

# Other arguments
parser.add_argument('--seed', type=int, default=2020, help='Seed')
parser.add_argument('--gpu', type=str, default='auto', # if 'auto', running three times in ipython will occupy three different GPUs.
                    help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least GPU memory")
parser.add_argument("--result", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/", help="Result directory")

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
# Training data
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
num_iter = args.num_iter
eval_interval = args.eval_interval
# Others
result = args.result
# Prediction or Evaluation
test_input1 = args.test_input1
test_input2 = args.test_input2
test_pred1 = args.test_pred1
test_pred2 = args.test_pred2
eval_model = args.eval_model
avg_pred_win = args.avg_pred_win

model_path = os.path.join(args.result, "model.ckpt") # Save the training model

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

def init_logger(file_name="", stream="stdout"):
    """ Initialize a logger to terminal and file at the same time. """
    import logging
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

set_seed(args.seed)
device = set_device(args.gpu)
print(f"Device: {device}")

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

def eval_input_function(xs1, xs2, labels1, labels2, batch_size, i):
    """
    Returns a batch of input features and labels for evaluation (development) set.

    Args:
        xs1 (array-like): Input data for the first stream, of shape (*, 257, 256)
        xs2 (array-like): Input data for the second stream, of shape (*, 257, 256)
        labels1 (array-like): Labels for the first stream, of shape (*, num_classes)
        labels2 (array-like): Labels for the second stream, of shape (*, num_classes)
        batch_size (int): The size of each batch.
        i (int): The index of the current batch.

    Returns:
        dict: A dictionary containing the i-th batch of input features and labels.
              Keys: 'x', 'x2', 'y', 'y2'. Values: NumPy arrays of type float32,
              of shape (batch_size, 257, 256) for input feature x and
              of shape (batch_size, num_classes) for target label y.
    """
    length = len(xs1)
    start_idx = (i % (length // batch_size)) * batch_size
    end_idx = start_idx + batch_size

    batch_data = {
        'x': xs1[start_idx:end_idx],
        'x2': xs2[start_idx:end_idx],
        'y': labels1[start_idx:end_idx],
        'y2': labels2[start_idx:end_idx]
    }

    return {key: np.array(value, dtype=np.float32) for key, value in batch_data.items()}

def input_function(xs1, xs2, labels1, labels2, batch_size, i, switch_streams=True, apply_random_shift=True):
    """
    Creates a minibatch of size `batch_size` from the input data for training.
    Each sample in the batch applies a new random shift.

    Args:
        xs1 (array-like): Input data for the first stream, of shape (*, 257, 256).
        xs2 (array-like): Input data for the second stream, of shape (*, 257, 256).
        labels1 (array-like): Labels corresponding to xs1, of shape (*, num_classes).
        labels2 (array-like): Labels corresponding to xs2, of shape (*, num_classes).
        batch_size (int): The size of each minibatch.
        i (int): The index of the current batch.
        switch_streams (bool): Whether to randomly switch the input-label pairing. Default is True.
        apply_random_shift (bool): Whether to apply random shifts to each sample in the batch. Default is True.

    Returns:
        dict: A dictionary containing the i-th minibatch of input features and labels.
              Keys: 'x', 'x2', 'y', 'y2'. Values: NumPy arrays of type float32,
              of shape (batch_size, 257, 256) for input feature x and
              of shape (batch_size, num_classes) for target label y.
    Note:
    Two optional randomizations:
    For the two-stream system, randomly feed data in two cases:
    data1 to stream1 and data2 to stream2 or
    data1 to stream2 and data2 to stream1,
    where each data include its input and label.

    Each sample in the batch applies a new random shift within 5 pixels.
    """
    length = len(xs1)
    start_idx = (i % (length // batch_size)) * batch_size
    end_idx = start_idx + batch_size

    # Randomly choose the input-label pairing if switch_streams is True
    if switch_streams and random.random() < 0.5:
        x1_batch, x2_batch = xs2[start_idx:end_idx], xs1[start_idx:end_idx]
        y1_batch, y2_batch = labels2[start_idx:end_idx], labels1[start_idx:end_idx]
    else:
        x1_batch, x2_batch = xs1[start_idx:end_idx], xs2[start_idx:end_idx]
        y1_batch, y2_batch = labels1[start_idx:end_idx], labels2[start_idx:end_idx]

    # Randomly roll each sample in the batch independently if apply_random_shift is True
    if apply_random_shift:
        x1_rolled = np.zeros_like(x1_batch)
        x2_rolled = np.zeros_like(x2_batch)
        for i in range(batch_size):
            ver_shift = random.randint(-5, 5)
            hor_shift = random.randint(-5, 5)
            x1_rolled[i] = np.roll(x1_batch[i], (ver_shift, hor_shift), axis=(0, 1))
            x2_rolled[i] = np.roll(x2_batch[i], (ver_shift, hor_shift), axis=(0, 1))
    else:
        x1_rolled = x1_batch
        x2_rolled = x2_batch

    batch_data = {
        'x': x1_rolled,
        'x2': x2_rolled,
        'y': y1_batch,
        'y2': y2_batch
    }

    return {key: np.array(value, dtype=np.float32) for key, value in batch_data.items()}

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

def train(model, optimizer, train_input1, train_input2, train_target_single1, train_target_single2,
          dev_input1, dev_input2, dev_target_single1, dev_target_single2,
          batch_size, num_iter, eval_interval, lr, model_path):
    """
    Function for training the network using the training and evaluation/development sets.

    Args:
        model (TwoStreamCNNModel): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_input1 (str): Path to the first training input file.
        train_input2 (str): Path to the second training input file.
        train_target_single1 (str): Path to the first training target file.
        train_target_single2 (str): Path to the second training target file.
        dev_input1 (str): Path to the first development input file.
        dev_input2 (str): Path to the second development input file.
        dev_target_single1 (str): Path to the first development target file.
        dev_target_single2 (str): Path to the second development target file.
        batch_size (int): Minibatch size for training.
        num_iter (int): Number of training iterations.
        eval_interval (int): Interval for evaluating on the development set.
        lr (float): Learning rate.
        model_path (str): Path to save the trained model.
    """
    logger = init_logger(os.path.join(os.path.dirname(model_path), "report.log")) # logger for training
    logger.info(args)

    # Load training and evaluation/development data from numpy files
    with open(train_input1, 'rb') as f:
        xs1 = np.load(f)
    with open(train_input2, 'rb') as f:
        xs2 = np.load(f)
    with open(train_target_single1, 'rb') as f:
        labels = np.load(f)
    with open(train_target_single2, 'rb') as f:
        labels2 = np.load(f)

    with open(dev_input1, 'rb') as f:
        eval_xs1 = np.load(f)
    with open(dev_input2, 'rb') as f:
        eval_xs2 = np.load(f)
    with open(dev_target_single1, 'rb') as f:
        eval_labels1 = np.load(f)
    with open(dev_target_single2, 'rb') as f:
        eval_labels2 = np.load(f)

    # Initialize the directory of path to save the model
    result_dir = os.path.dirname(model_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logger.info('Training new model')

    length = xs1.shape[0]
    s = list(range(length))
    accuracies = []
    for i in range(num_iter):
        numb = i % (length // batch_size)
        if numb == 0:
            # Shuffle training data after a full epoch
            random.shuffle(s)
            labels = labels[s]
            labels2 = labels2[s]
            xs1 = xs1[s]
            xs2 = xs2[s]

        # Get the input batch using the input_function
        inputs = input_function(xs1, xs2, labels, labels2, batch_size, i)
        # Run a training step and calculate accuracy
        _, _, loss, accurs = model(torch.from_numpy(inputs['x']).to(device),
                                   torch.from_numpy(inputs['x2']).to(device),
                                   torch.from_numpy(inputs['y']).argmax(dim=1).to(device),  # onehot to label
                                   torch.from_numpy(inputs['y2']).argmax(dim=1).to(device))
        accuracies.extend([1 if val else 0 for val in accurs])

        model.train()

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
        optimizer.step()
        if np.isnan(loss.item()): raise ValueError("NaN detected")

        if i % eval_interval == 0:
            # Print training accuracy and evaluate on the development set
            logger.info(f"Step {i} Train accuracy: {np.mean(accuracies):.4f}")
            logger.info(f"Decaying the learning rate from '{lr:.8f}' to '{lr*0.97:.8f}'")
            lr *= 0.97  # Decay the learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr

            accuracies = []
            for k in range(eval_xs1.shape[0] // batch_size):
                # Get the evaluation input batch using the eval_input_function
                inputs = eval_input_function(eval_xs1, eval_xs2, eval_labels1, eval_labels2, batch_size, k)
                # Run the model to get the accuracy on the evaluation batch
                _, _, loss, accurs = model(torch.from_numpy(inputs['x']).to(device),
                                           torch.from_numpy(inputs['x2']).to(device),
                                           torch.from_numpy(inputs['y']).argmax(dim=1).to(device),  # onehot to label
                                           torch.from_numpy(inputs['y2']).argmax(dim=1).to(device))
                accuracies.extend([1 if val else 0 for val in accurs])
            logger.info(f"Eval accuracy: {np.mean(accuracies):.4f}")
            accuracies = []
            # Save the trained model
            checkpoint = {
                "batch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved: {model_path}")

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
    logger = init_logger(os.path.join(os.path.dirname(predictions_file1), "report.log")) # logger for testing
    logger.info(args)
    # Make output directories for predicted files
    for output_dir in [os.path.dirname(predictions_file1), os.path.dirname(predictions_file2)]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

    start_time = datetime.datetime.now()
    train(model, optimizer, train_input1, train_input2, train_target_single1, train_target_single2, dev_input1, dev_input2, dev_target_single1, dev_target_single2,
          batch_size, num_iter, eval_interval, lr, model_path)
    duration = datetime.datetime.now() - start_time
    print(f'Time taken to complete the training: {duration.seconds // 3600:02}:{(duration.seconds // 60) % 60:02}:{duration.seconds % 60:02}')

    # set_seed(args.seed)
    # predict(model, test_input1, test_input2, test_pred1, test_pred2, model_path=eval_model, avg_pred_win=avg_pred_win, batch_size=batch_size)
