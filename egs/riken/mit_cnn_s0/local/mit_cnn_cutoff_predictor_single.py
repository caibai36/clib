'''creates text files out of predictions for single animal at a time (9 node output layer)'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs

truth_values=['cha','chi','ek','ph','ts','tr','trph','tw','noise']

#returns the top prediction if has confidence higher than cutoff, otherwise returns noise
def get_prediction(preds,cutoff):
    max_pred=np.argmax(preds)
    #if tr or ph, also count in confidence for trph
    if max_pred in [3,5]:
        if preds[max_pred]+preds[6]<cutoff:
            return(8)
        else:
            return(max_pred)
    
    #if trph sum probs for tr, trph and ph
    elif max_pred==6:
        if preds[max_pred]+preds[3]+preds[5]<cutoff:
            return(8)
        else:
            return(max_pred)
    else:    
        if preds[max_pred]<cutoff:
            return(8)
        else:
            return(max_pred)
 
def predict(pred_data,predictions_file1,cutoff):
    fil1=open(pred_data,'rb')
    predictions=np.load(fil1)
    fil1.close()
    #Uses utf-8 for Audacity compability
    save_fil=codecs.open(predictions_file1,'w', 'utf-8')
    for i in range(len(predictions)):
        numb=get_prediction(predictions[i],cutoff)
        if numb!=8:
            save_fil.write('{:.3f}\t{:.3f}\t{}\n'.format((i)*0.05+0.325, (i+1)*0.05+0.325, truth_values[numb]))
        
    print('Done!')
    save_fil.close()


# if __name__=='__main__':
#   for num in [0,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
#       predict('Data/pred_20161219_Athos_model_small.npy',
#               'results/20161219_Athos_model_small_{}%.txt'.format(int(100*num)),cutoff=num)
#       predict('Data/pred_20161219_Porthos_model_small.npy',
#               'results/20161219_Porthos_model_small_{}%.txt'.format(int(100*num)),cutoff=num)

import os
import argparse

default_pred_files=["exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos.npy", "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos.npy"]
parser = argparse.ArgumentParser(description="Cutoff the predictions (reference: 'cutoff_predictor_single.py' from https://marmosetbehavior.mit.edu/).")
parser.add_argument("--cutoffs", type=float, default=[0.7, 0.8], nargs="+", help="values of cutoff. e.g., --cutoffs 0.7 0.8")
parser.add_argument("--pred_files", type=str, default=default_pred_files, nargs="+", help='a sequence of predictions that need to be cut off')
parser.add_argument("--out_dir", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval", help="output directory to store predictions after cut off")
args = parser.parse_args()

for cutoff in args.cutoffs:
    for pred_file in args.pred_files:
        file_name = os.path.splitext(os.path.basename(pred_file))[0]
        cutoff_file_name = file_name + "_cutoff{}".format(cutoff) + ".txt"
        if int(cutoff) == cutoff: cutoff_file_name = file_name + "_cutoff{}".format(int(cutoff)) + ".txt" # Avoid float converting cutoff 0 into cutoff 0.0
        cutoff_file = os.path.join(args.out_dir, cutoff_file_name)

        print("Pred file: {}".format(pred_file))
        print("Cutoff file: {}".format(cutoff_file))
        predict(pred_file, cutoff_file, cutoff=cutoff)
