# Implemented by bin-wu at 14:22 on 4 April 2024 from the MIT's implementation
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Convert a multi-hot target into two one-hot targets of an animal pair for the training and development sets (reference: 'data_converter_single2.py' from https://marmosetbehavior.mit.edu/)\n\nOpen two files with multi-hot targets from NumPy's .npy format. The index dict of ({'cha':0,'chi':1,'ek':2,'ph':3,'ts':4,'tr':5,'trph':6,'tw':7,'noise':8,'tw2':9,'trph2':10,'tr2':11,'ts2':12,'ph2':13,'ek2':14,'chi2':15,'cha2':16}) is symmetric from 8 ('noise'). The left indices from 0 to 7 are from the first animal and the right indices from 9 to 16 are from the second animal. Split the 17-dimensional multi-hot representation into two 8-dimensional one-hot representations. Each one with a noise at index 8. When there is no call of an animal, add noise to the one-hot representation. Save the two one-hot representations.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--train_target_multi", type=str, default="exp/data/mit_sample/train_target_multi",
                    help="The input file of the multi-hot target of an animal pair for the training set")
parser.add_argument("--dev_target_multi", type=str, default="exp/data/mit_sample/dev_target_multi",
                    help="The input file of the multi-hot target of an animal pair for the development set")
parser.add_argument("--train_target_single1", type=str, default="exp/data/mit_sample/train_target_single1",
                    help="The output file of the one-hot target of the first animal for the training set")
parser.add_argument("--train_target_single2", type=str, default="exp/data/mit_sample/train_target_single2",
                    help="The output file of the one-hot target of the second animal for the training set")
parser.add_argument("--dev_target_single1", type=str, default="exp/data/mit_sample/dev_target_single1",
                    help="The output file of the one-hot target of the first animal for the development set")
parser.add_argument("--dev_target_single2", type=str, default="exp/data/mit_sample/dev_target_single2",
                    help="The output file of the one-hot target of the second animal for the development set")

args = parser.parse_args()
print(args)

# Load training and development data labels
with open(args.train_target_multi, 'rb') as f:
    train_labels = np.load(f)
with open(args.dev_target_multi, 'rb') as f:
    dev_labels = np.load(f)

# Split multi-hot targets into two one-hot targets
def split_labels(labels):
    labels1 = []
    labels2 = []
    for label in labels:
        # Extract the first 9 elements for the first animal
        label1 = label[:9]
        # Extract the last 9 elements for the second animal and flip the order
        label2 = np.flipud(label)[:9]
        
        # If there is no call by the first animal, set the noise label to 1
        if np.sum(label1) == 0:
            label1 = np.zeros(9)
            label1[-1] = 1
        
        # If there is no call by the second animal, set the noise label to 1
        if np.sum(label2) == 0:
            label2 = np.zeros(9)
            label2[-1] = 1
        
        labels1.append(label1)
        labels2.append(label2)
    
    # Convert the lists of labels into NumPy arrays
    return np.array(labels1), np.array(labels2)

# Split the training and development labels into two one-hot targets
train_labels1, train_labels2 = split_labels(train_labels)
dev_labels1, dev_labels2 = split_labels(dev_labels)

# Save one-hot targets
data_sets = [
    (args.train_target_single1, train_labels1),
    (args.train_target_single2, train_labels2),
    (args.dev_target_single1, dev_labels1),
    (args.dev_target_single2, dev_labels2)
]

for filename, data in data_sets:
    with open(filename, 'wb') as f:
        np.save(f, data)
