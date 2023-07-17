'''splits the combined labels into separate labels for each animal'''
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Convert an onehot target into two onehot targets of an animal pair for the training and development sets (reference: 'data_converter_single2.py' from https://marmosetbehavior.mit.edu/).")
parser.add_argument("--train_target_multi", type=str, default="exp/data/mit_sample/train_target_multi",
                    help="The input file of the onehot target of an animal pair for the training set")
parser.add_argument("--dev_target_multi", type=str, default="exp/data/mit_sample/dev_target_multi",
                    help="The input file of the onehot target of an animal pair for the development set")
parser.add_argument("--train_target_single1", type=str, default="exp/data/mit_sample/train_target_single1",
                    help="The output file of the onehot target of the first animal for the training set")
parser.add_argument("--train_target_single2", type=str, default="exp/data/mit_sample/train_target_single2",
                    help="The output file of the onehot target of the second animal for the training set")
parser.add_argument("--dev_target_single1", type=str, default="exp/data/mit_sample/dev_target_single1",
                    help="The output file of the onehot target of the first animal for the development set")
parser.add_argument("--dev_target_single2", type=str, default="exp/data/mit_sample/dev_target_single2",
                    help="The output file of the onehot target of the second animal for the development set")

args = parser.parse_args()
print(args)

#training data labels
fil1=open(args.train_target_multi,'rb')
old_train_labels=np.load(fil1)
fil1.close()
#eval data labels
fil2=open(args.dev_target_multi,'rb')
old_eval_labels=np.load(fil2)
fil2.close()

new_train_labels=[]
new_train_labels2=[]
for i in range(len(old_train_labels)):
    init_y=old_train_labels[i,:9]
    #if no call by animal1, add noise label
    if np.sum(init_y)==0:
        new_train_labels.append([0,0,0,0,0,0,0,0,1])
    else:
        new_train_labels.append(init_y)

    #if no call by animal2, use noise label, otherwise use flipped
    #version of the last 9 labels to make it the same format as animal1
    init_y2=np.flipud(old_train_labels[i])[:9]
    if np.sum(init_y2)==0:
        new_train_labels2.append([0,0,0,0,0,0,0,0,1])
    else:
        new_train_labels2.append(init_y2)


new_eval_labels=[]
new_eval_labels2=[]
for i in range(len(old_eval_labels)):
    init_y=old_eval_labels[i,:9]
    if np.sum(init_y)==0:
        new_eval_labels.append([0,0,0,0,0,0,0,0,1])
    else:
        new_eval_labels.append(init_y)

    init_y2=np.flipud(old_eval_labels[i])[:9]
    if np.sum(init_y2)==0:
        new_eval_labels2.append([0,0,0,0,0,0,0,0,1])
    else:
        new_eval_labels2.append(init_y2)

#train labels1
save_fil1=open(args.train_target_single1,'wb')
np.save(save_fil1,new_train_labels)
save_fil1.close()
#train labels2
save_fil2=open(args.train_target_single2,'wb')
np.save(save_fil2,new_train_labels2)
save_fil2.close()
#eval labels1
save_fil3=open(args.dev_target_single1,'wb')
np.save(save_fil3,new_eval_labels)
save_fil3.close()
#eval labels2
save_fil4=open(args.dev_target_single2,'wb')
np.save(save_fil4,new_eval_labels2)
save_fil4.close()
