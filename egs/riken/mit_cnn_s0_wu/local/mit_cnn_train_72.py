# Implemented by bin-wu at 19:23 on 5 April 2024 from the MIT's implementation

import os
import datetime

import math
import random
import argparse

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Argument parser
parser = argparse.ArgumentParser(description=("Train or evaluate MIT CNN 72. (Reference: 'new_train_72.py' from https://marmosetbehavior.mit.edu/, supporting TensorFlow 2)"))

# Data arguments
parser.add_argument("--train_input1", type=str, default="exp/data/mit_sample/train_input1", help="Training set's first stream input")
parser.add_argument("--train_input2", type=str, default="exp/data/mit_sample/train_input2", help="Training set's second stream input")
parser.add_argument("--train_target_single1", type=str, default="exp/data/mit_sample/train_target_single1", help="Training set's first stream target label")
parser.add_argument("--train_target_single2", type=str, default="exp/data/mit_sample/train_target_single2", help="Training set's second stream target label")
parser.add_argument("--dev_input1", type=str, default="exp/data/mit_sample/dev_input1", help="Development set's first stream input")
parser.add_argument("--dev_input2", type=str, default="exp/data/mit_sample/dev_input2", help="Development set's second stream input")
parser.add_argument("--dev_target_single1", type=str, default="exp/data/mit_sample/dev_target_single1", help="Development set's first stream target label")
parser.add_argument("--dev_target_single2", type=str, default="exp/data/mit_sample/dev_target_single2", help="Development set's second stream target label")
parser.add_argument("--test_input1", type=str, default="exp/data/mit_sample/test_input1_Athos", help="Testing set's first stream input")
parser.add_argument("--test_input2", type=str, default="exp/data/mit_sample/test_input2_Porthos", help="Testing set's second stream input")
parser.add_argument("--test_pred1", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos", help="Test set's first stream predicted label probabilities")
parser.add_argument("--test_pred2", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos", help="Test set's second stream predicted label probabilities")

# Model arguments
parser.add_argument("--batch_size", type=int, default=25, help="Batch size for the dataloader")
parser.add_argument("--dropout_rate", type=float, default=0.4, help="Drop rate of dropout layer")
parser.add_argument("--eval_model", type=str, default="", help="Model path for prediction or evaluation")

# Optimizer arguments
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate of Adam optimizer")
parser.add_argument("--epsilon", type=float, default=0.001, help="Epsilon of Adam optimizer for numerical stability")

# Training arguments
parser.add_argument("--num_iter", type=int, default=2601, help="Number of iterations")
parser.add_argument("--eval_interval", type=int, default=200, help="Evaluate the development set every x iterations")

# Evaluation arguments
parser.add_argument("--avg_pred_win", type=int, default=5, help="Collect predicted probabilities by averaging across x consecutive predictions")

# Other arguments
parser.add_argument("--result", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/train/", help="Result directory")

# Parse arguments
args = parser.parse_args()

# Check if training or evaluation
is_training = not args.eval_model
if not os.path.exists(args.result) and is_training:
    os.makedirs(args.result)

model_path = os.path.join(args.result, "model.ckpt")

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

# Print the arguments for verification
print(args)

truth_values=['cha','chi','ek','ph','ts','tr','trph','tw','noise']
tf.logging.set_verbosity(tf.logging.INFO)

def eval_input_function(xs1, xs2, labels1, labels2, batch_size, i):
    """
    Returns a batch of input features and labels for evaluation (development) set.

    Args:
        xs1 (array-like): Input data for the first stream
        xs2 (array-like): Input data for the second stream
        labels1 (array-like): Labels for the first stream
        labels2 (array-like): Labels for the second stream
        batch_size (int): The size of each batch.
        i (int): The index of the current batch.

    Returns:
        dict: A dictionary containing the i-th batch of input features and labels.
              Keys: 'x', 'x2', 'y', 'y2'. Values: NumPy arrays of type float32.
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
        xs1 (array-like): Input data for the first stream.
        xs2 (array-like): Input data for the second stream.
        labels1 (array-like): Labels corresponding to xs1.
        labels2 (array-like): Labels corresponding to xs2.
        batch_size (int): The size of each minibatch.
        i (int): The index of the current batch.
        switch_streams (bool): Whether to randomly switch the input-label pairing. Default is True.
        apply_random_shift (bool): Whether to apply random shifts to each sample in the batch. Default is True.

    Returns:
        dict: A dictionary containing the minibatch of input features and labels.
              Keys: 'x', 'x2', 'y', 'y2'. Values: NumPy arrays of type float32.
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
        xs1 (array-like): Input spectral segments for the first stream.
        xs2 (array-like): Input spectral segments for the second stream.
        i (int): The index of the current spectral segment.
        window_size (int): The size of each element (default: 256).
        step_size (int): The step size for sliding the window (default: 26).

    Returns:
        dict: A dictionary containing the minibatch of input features.
              Keys: 'x', 'x2'. Values: NumPy arrays of shape (50, 257, 256).

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

def create_model(mode):
    """
    Creates the two stream convolutional neural net with 4 blocks of conv-conv-maxpool.

    Args:
        mode: A string indicating the mode of operation ('train' or 'predict').

    Returns:
        x: Placeholder for the first input spectrogram.
        x2: Placeholder for the second input spectrogram.
        y: Placeholder for the first set of labels.
        y2: Placeholder for the second set of labels.
        probabilities: Softmax probabilities for the first output.
        probabilities2: Softmax probabilities for the second output.
        accuracy: Logical AND of the equality between predicted and correct classes for both outputs.
        lrate: Placeholder for the learning rate.
        train_step: Operation to perform a training step.
        loss: Combined loss from both outputs.
    """
    # Input Layer
    x = tf.placeholder(tf.float32, [None, 257, 256])
    input_layer = tf.reshape(x, [-1, 257, 256, 1])
    pool0 = tf.layers.max_pooling2d(inputs=input_layer, pool_size=[2, 2], strides=2)

    # First convolutional stream
    conv1 = tf.layers.conv2d(
        inputs=pool0,
        filters=16,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=16,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=32,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=64,
        kernel_size=[3, 3], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    conv7 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=64,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2)

    # Second convolutional stream for second input
    x2 = tf.placeholder(tf.float32, [None, 257, 256])
    input_layer2 = tf.reshape(x2, [-1, 257, 256, 1])

    pool02 = tf.layers.max_pooling2d(inputs=input_layer2, pool_size=[2, 2], strides=2)

    conv12 = tf.layers.conv2d(
        inputs=pool02,
        filters=16,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv22 = tf.layers.conv2d(
        inputs=conv12,
        filters=16,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool12 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2)

    conv32 = tf.layers.conv2d(
        inputs=pool12,
        filters=32,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv42 = tf.layers.conv2d(
        inputs=conv32,
        filters=32,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool22 = tf.layers.max_pooling2d(inputs=conv42, pool_size=[2, 2], strides=2)

    conv52 = tf.layers.conv2d(
        inputs=pool22,
        filters=64,
        kernel_size=[3, 3], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv62 = tf.layers.conv2d(
        inputs=conv52,
        filters=64,
        kernel_size=[3, 3], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool32 = tf.layers.max_pooling2d(inputs=conv62, pool_size=[2, 2], strides=2)

    conv72 = tf.layers.conv2d(
        inputs=pool32,
        filters=64,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    conv82 = tf.layers.conv2d(
        inputs=conv72,
        filters=64,
        kernel_size=[5, 5], strides=1,
        padding="same",
        activation=tf.nn.relu)
    pool42 = tf.layers.max_pooling2d(inputs=conv82, pool_size=[2, 2], strides=2)

    # Combines the two convolutional streams and reshapes them into 1D (excluding batch dimension)
    final_pool_flat = tf.concat(
        [tf.reshape(pool4, [-1, 8 * 8 * 64]),
         tf.reshape(pool42, [-1, 8 * 8 * 64])],
        axis=1)

    # Adds a 1024 fully connected layer with dropout
    dense = tf.layers.dense(inputs=final_pool_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, training=mode == 'train')

    # Creates the two output layers (logits, logits2)
    logits = tf.layers.dense(inputs=dropout, units=9)
    logits2 = tf.layers.dense(inputs=dropout, units=9)

    # Placeholders for labels
    y = tf.placeholder(tf.float32, [None, 9])
    y2 = tf.placeholder(tf.float32, [None, 9])

    # Computes the predicted classes and accuracy
    classes = tf.argmax(logits, axis=1)
    classes2 = tf.argmax(logits2, axis=1)
    correct = tf.argmax(y, axis=1)
    correct2 = tf.argmax(y2, axis=1)
    accuracy = tf.logical_and(tf.equal(classes, correct), tf.equal(classes2, correct2))

    # Computes the softmax probabilities for both outputs
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")
    probabilities2 = tf.nn.softmax(logits2, name="softmax_tensor")

    # Computes the loss for both outputs
    loss1 = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    loss = loss1 + tf.losses.softmax_cross_entropy(onehot_labels=y2, logits=logits2)

    # Placeholder for learning rate
    lrate = tf.placeholder(tf.float32, None)

    # Creates an optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate, epsilon=epsilon)

    # Updates batch normalization parameters (if used)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss)

    return x, x2, y, y2, probabilities, probabilities2, accuracy, lrate, train_step, loss

def main(xs1=None, xs2=None, batch_size=10, mode='predict', model_dir='Models/model.ckpt', lr=lr, avg_pred_win=avg_pred_win):
    """
    Main function for using the network.

    Args:
        xs1 (numpy.ndarray): Input data for the first stream (default: None).
        xs2 (numpy.ndarray): Input data for the second stream (default: None).
        batch_size (int): Minibatch size for training (default: 10).
        mode (str): Mode of operation, either 'train' or 'predict' (default: 'predict').
        model_dir (str): Path to save/load the model (default: 'Models/model.ckpt').
        lr (float): Learning rate (default: lr).
        avg_pred_win (int): Size of the averaging window for predictions (default: 5).

    Returns:
        tuple: A tuple containing the averaged predictions for the two streams (only in 'predict' mode).
    """
    if mode == 'train':
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

    # Create the model using the create_model function
    x, x2, y, y2, probabilities, probabilities2, accuracy, lrate, train_step, loss = create_model(mode)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if mode == 'predict':
            # Restore the saved model for prediction
            saver.restore(sess, model_dir)
            print('Model restored')
        else:
            # Initialize variables for training a new model
            print('Training new model')
            sess.run(tf.global_variables_initializer())

        if mode == 'predict':
            # Perform predictions on the input data
            preds_list = []
            preds_list2 = []
            predictions = []
            predictions2 = []
            print('Predicting')
            length = min(xs1.shape[0], xs2.shape[0])
            num_batches = length
            for i in range(num_batches - 1):
                # Get the input batch using the pred_input_function
                inputs = pred_input_function(xs1, xs2, i)
                # Run the model to get the predictions for the current batch
                preds, preds2 = sess.run([probabilities, probabilities2],
                                          feed_dict={x: inputs['x'], x2: inputs['x2']})
                # Append the predictions to the preds_list and preds_list2
                preds_list.extend(preds)
                preds_list2.extend(preds2)
                if (i + 1) % 25 == 0:
                    print(f"Processed {i + 1}/{num_batches} batches")

            # Average predictions across consecutive windows
            for i in range(len(preds_list) - (avg_pred_win - 1)):
                # Calculate the mean predictions for the current window
                mean_preds = np.mean(preds_list[i:i + avg_pred_win], axis=0)
                mean_preds2 = np.mean(preds_list2[i:i + avg_pred_win], axis=0)
                # Append the mean predictions to the final predictions lists
                predictions.append(mean_preds)
                predictions2.append(mean_preds2)

            print(f"Predictions shape: {np.shape(predictions)}")
            return predictions, predictions2

        elif mode == 'train':
            # Train the model
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
                _, accurs = sess.run([train_step, accuracy],
                                     feed_dict={x: inputs['x'], x2: inputs['x2'],
                                                y: inputs['y'], y2: inputs['y2'], lrate: lr})
                accuracies.extend([1 if val else 0 for val in accurs])

                if i % eval_interval == 0:
                    # Print training accuracy and evaluate on the development set
                    print(f"Step {i} Train accuracy: {np.mean(accuracies):.4f}")
                    lr *= 0.97  # Decay the learning rate
                    accuracies = []
                    for k in range(eval_xs1.shape[0] // batch_size):
                        # Get the evaluation input batch using the eval_input_function
                        inputs = eval_input_function(eval_xs1, eval_xs2, eval_labels1, eval_labels2, batch_size, k)
                        # Run the model to get the accuracy on the evaluation batch
                        accurs = sess.run(accuracy, feed_dict={x: inputs['x'], x2: inputs['x2'],
                                                               y: inputs['y'], y2: inputs['y2']})
                        accuracies.extend([1 if val else 0 for val in accurs])
                    print(f"Eval accuracy: {np.mean(accuracies):.4f}")
                    accuracies = []
                    # Save the trained model
                    savepath = saver.save(sess, model_dir)
                    print(f"Model saved: {savepath}")

def train(model_dir):
     """
     Function for training with the network for the training and evaluation/development sets.

     Args:
        model_dir (str): Path to save the trained model.
     """
     main(batch_size=batch_size, mode='train', model_dir=model_dir)

def predict(pred_data1, pred_data2, predictions_file1, predictions_file2, model_dir):
    """
    Function for predicting with the network for the testing set.

    Args:
        pred_data1 (str): Path to the first file containing spectrogram arrays for prediction.
        pred_data2 (str or None): Path to the second input file from the same session.
                                  If None, an array of zeros will be used as the second input.
        predictions_file1 (str): Path to save the predicted probabilities for the first input.
        predictions_file2 (str): Path to save the predicted probabilities for the second input.
        model_dir (str): Path to the saved model to be used for prediction.

    Returns:
        None
    """
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

    # Perform predictions using the main function
    predictions1, predictions2 = main(predict_x1, predict_x2, batch_size=batch_size, mode='predict',
                                      model_dir=model_dir)

    # Save the predictions to files
    np.save(predictions_file1, predictions1)
    np.save(predictions_file2, predictions2)

if __name__=='__main__':
     start_time = datetime.datetime.now()
     train(model_path)
     duration = datetime.datetime.now() - start_time
     print(f'Time taken to complete the training: {duration.seconds // 3600:02}:{(duration.seconds // 60) % 60:02}:{duration.seconds % 60:02}')
     # predict(test_input1, test_input2, test_pred1, test_pred2, model_dir=eval_model)
     # print(f"Used the model: {eval_model}")
