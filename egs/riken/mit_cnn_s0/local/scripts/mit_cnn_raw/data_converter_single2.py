'''splits the combined labels into separate labels for each animal'''
import numpy as np

#training data labels
fil1=open('Data/train_ys_multi','rb')
old_train_labels=np.load(fil1)
fil1.close()
#eval data labels
fil2=open('Data/eval_ys_multi','rb')
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
save_fil1=open('Data/train_ys_single','wb')
np.save(save_fil1,new_train_labels)
save_fil1.close()
#train labels2
save_fil2=open('Data/train_ys_single2','wb')
np.save(save_fil2,new_train_labels2)
save_fil2.close()
#eval labels1
save_fil3=open('Data/eval_ys_single','wb')
np.save(save_fil3,new_eval_labels)
save_fil3.close()
#eval labels2
save_fil4=open('Data/eval_ys_single2','wb')
np.save(save_fil4,new_eval_labels2)
save_fil4.close()
