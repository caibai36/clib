'''Used for training and predicting a convolutional neural net with two 9 node
output layers. Does not use batch normalization'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import pickle
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

truth_values=['cha','chi','ek','ph','ts','tr','trph','tw','noise']
tf.logging.set_verbosity(tf.logging.INFO)

def input_function(xs1,xs2, labels,labels2, batch_size,i):
    """An input function for training,creates a minibatch of batch_size from input data"""
    length=np.shape(xs1)[0]
    #numb gives the number of next batch to be used given i
    numb=i%(length//batch_size)
    new_features={}
    #randomizes which input goes to which channel
    if random.random()>0.5:
        change={'x':'x','x2':'x2'}
        new_features['y']=labels[batch_size*numb:batch_size*(numb+1)]
        new_features['y2']=labels2[batch_size*numb:batch_size*(numb+1)]
    else:
        change={'x':'x2','x2':'x'}
        new_features['y']=labels2[batch_size*numb:batch_size*(numb+1)]
        new_features['y2']=labels[batch_size*numb:batch_size*(numb+1)]
    for key in ['x','x2']:
        new_features[key]=[]

    '''randomly rolls each pair of spectrograms in the batch for up to 5 pixels
    horizontally and vertically. Roll amount is same for both spectrograms'''
    
    for i in range(batch_size*numb,batch_size*(numb+1)):
        ver_shift=random.randint(-5,5)
        hor_shift=random.randint(-5,5)
        a=np.roll(xs1[i],(ver_shift,hor_shift),axis=(0,1))
        new_features[change['x']].append(a)
        
        a2=np.roll(xs2[i],(ver_shift,hor_shift),axis=(0,1))
        new_features[change['x2']].append(a2)

    for key in new_features.keys():
        new_features[key]=np.array(new_features[key],dtype=np.float32)
    return new_features

def eval_input_function(xs1,xs2, labels,labels2, batch_size,i):
    """An input function for evaluation, same as input_function() but doesn't use
    randomization"""
    length=np.shape(xs1)[0]
    numb=i%(length//batch_size)
    new_features={}
    new_features['y']=labels[batch_size*numb:batch_size*(numb+1)]
    new_features['y2']=labels2[batch_size*numb:batch_size*(numb+1)]
    for key in ['x','x2']:
        new_features[key]=[]
    for i in range(batch_size*numb,batch_size*(numb+1)):
        new_features['x'].append(xs1[i])
        new_features['x2'].append(xs2[i])

    for key in new_features.keys():
        new_features[key]=np.array(new_features[key],dtype=np.float32)
    return (new_features)


def pred_input_function(xs1,xs2, i):
    """An input function for predictions, makes a minibatch of 50
    spectrograms with step size of 50ms"""
    
    new_features={}
    for key in ['x','x2']:
        new_features[key]=[]
    '''splits 257x1299 array into 50 separate 257x256 pieces, 50ms stepsize translates to
    1299/50~26 pixels step size. First 41 pieces can be sliced from the first spectrogram array
    while the other 9 have to be concatenated with the first parts of the next array'''
    for j in range(41):
      start=26*j
      a=xs1[i,:,start:start+256]
      new_features['x'].append(a)
      
      a=xs2[i,:,start:start+256]
      new_features['x2'].append(a)
      
    for k in range(9):  
      start=26*(k+41)
      init_a=xs1[i,:,start:]
      last=256-np.shape(init_a)[1]
      a=np.concatenate([init_a,xs1[i+1,:,:last]],axis=1)
      new_features['x'].append(a)
      
      init_a=xs2[i,:,start:]
      a=np.concatenate([init_a,xs2[i+1,:,:last]],axis=1)
      new_features['x2'].append(a)
    for key in new_features.keys():
      new_features[key]=np.array(new_features[key],dtype=np.float32)
    return new_features

def main(xs1=None,xs2=None, batch_size=10, mode='predict', model_dir='Models/model.ckpt'):
  '''the main function for using the network,
  -xs1 and xs2 are used to feed input files when in prediction mode,
  -batch_size refers to the minibatch size to be used when training,
  higher batch_size makes the network train faster but also use more memory, batch_size for predictions is
  hard coded to 50 due to more complicated data structure for performance gains.
  -mode tells whether the network is predicting or training, should be one of 'train' and 'predict'
  -model_dir is the path to where the model should be saved/loaded from'''
  #loads training data, done here to minimize memory usage
  if mode=='train':
      fil_tr_x1=open('Data/train_xs1','rb')
      xs1=np.load(fil_tr_x1)
      fil_tr_x1.close()
      fil_ev_x1=open('Data/eval_xs1','rb')
      eval_xs1=np.load(fil_ev_x1)
      fil_ev_x1.close()

      fil_tr_x2=open('Data/train_xs2','rb')
      xs2=np.load(fil_tr_x2)
      fil_tr_x2.close()
      fil_ev_x2=open('Data/eval_xs2','rb')
      eval_xs2=np.load(fil_ev_x2)
      fil_ev_x2.close()

      fil_tr_y=open('Data/train_ys_single','rb')
      labels=np.load(fil_tr_y)
      fil_tr_y.close()
      fil_ev_y=open('Data/eval_ys_single','rb')
      eval_labels=np.load(fil_ev_y)
      fil_ev_y.close()

      fil_tr_y2=open('Data/train_ys_single2','rb')
      labels2=np.load(fil_tr_y2)
      fil_tr_y2.close()
      fil_ev_y2=open('Data/eval_ys_single2','rb')
      eval_labels2=np.load(fil_ev_y2)
      fil_ev_y2.close()
    
  """Creates the two stream convolutional neural net with 4 blocks of conv-conv-maxpool"""
  # Input Layer
  x = tf.placeholder(tf.float32, [None, 257, 256])
  input_layer=tf.reshape(x,[-1,257,256,1])
  pool0 = tf.layers.max_pooling2d(inputs=input_layer, pool_size=[2, 2], strides=2)
  
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
  
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)#64,52
  
  
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
  
  pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)  #128,105
  
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
  
  #Second convolutional stream for second input
  x2=tf.placeholder(tf.float32,[None,257,256])
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
  
  # Combines the two convolutional nets and reshapes them into 1D(excluding batch dimension)
  final_pool_flat = tf.concat([tf.reshape(pool4, [-1, 8 * 8 * 64]),
                               tf.reshape(pool42, [-1, 8 * 8 * 64])],axis=1)
  #adds a 1024 fully connected layer with droupout
  dense = tf.layers.dense(inputs=final_pool_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == 'train')
  #Creates the two output layers(logits, logits2)
  logits = tf.layers.dense(inputs=dropout, units=9)
  logits2= tf.layers.dense(inputs=dropout, units=9)
  #placeholders for labels
  y=tf.placeholder(tf.float32,[None,9])
  y2=tf.placeholder(tf.float32,[None,9])
  
  classes=tf.argmax(logits, axis=1)
  classes2=tf.argmax(logits2, axis=1)
  correct=tf.argmax(y, axis=1)
  correct2=tf.argmax(y2, axis=1)
  accuracy=tf.logical_and(tf.equal(classes,correct),tf.equal(classes2,correct2))
  
  probabilities=tf.nn.softmax(logits, name="softmax_tensor")
  probabilities2=tf.nn.softmax(logits2, name="softmax_tensor")
  
  loss1 = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
  loss=loss1+tf.losses.softmax_cross_entropy(onehot_labels=y2, logits=logits2)
  lr=0.0003
  lrate=tf.placeholder(tf.float32,None)
  optimizer = tf.train.AdamOptimizer(learning_rate=lrate,epsilon=0.001)

  #needed if using batch normalization to make sure it works properly
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = optimizer.minimize(loss)

  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  #if in prediction mode will use a saved model but if in train mode will train a new model instead
  if mode=='predict':
      saver.restore(sess, model_dir)
      print('Model restored')
  else:
      print('Training new model')
      tf.global_variables_initializer().run()
  
  if mode == 'predict':
    preds_list=[]
    preds_list2=[]
    predictions=[]
    predictions2=[]
    print('Predicting')
    length=min(np.shape(xs1)[0],np.shape(xs2)[0])
    num_batches=length
    for i in range(num_batches-1):
        inputs=pred_input_function(xs1,xs2, i)
        preds,preds2=sess.run([probabilities,probabilities2],feed_dict={x:inputs['x'],x2:inputs['x2']})
        for j in range(len(preds)):
            preds_list.append(preds[j])
            preds_list2.append(preds2[j])
        if i%25==0:
            print('{}/{}'.format(i,num_batches))
    '''After getting predictions for each timestep, makes final lists(predictions,predictions2)
    by averaging across 5 consecutive predictions,(step size of 1)'''
    
    for i in range(len(preds_list)-4):
        mean_preds=np.mean(preds_list[i:i+5],axis=0)
        mean_preds2=np.mean(preds_list2[i:i+5],axis=0)
        predictions.append(mean_preds)
        predictions2.append(mean_preds2)
        
    print(np.shape(predictions))
    tf.reset_default_graph()
    sess.close()
    return predictions,predictions2
  
  
  if mode=='train':
    
    length=np.shape(xs1)[0]
    s=[]
    accuracies=[]
    for j in range(length):
        s.append(j)
    for i in range(2601):
        numb=i%(length//batch_size)
        #shuffles training data after a full epoch
        if numb==0:
            random.shuffle(s)
            labels=labels[s]
            labels2=labels2[s]
            xs1=xs1[s]
            xs2=xs2[s]
            
        inputs=input_function(xs1,xs2, labels,labels2, batch_size,i)
        #trains the model while keeping track of accuracy
        _,accurs=sess.run([train_step,accuracy],feed_dict={x:inputs['x'],x2:inputs['x2'],
                                                           y:inputs['y'],y2:inputs['y2'],lrate:lr})
        for value in accurs:
            if value==True:
                accuracies.append(1)
            else:
                accuracies.append(0)
        #prints training accuracy and evaluates evaluation accuracy every 2000 steps
        #also multiplies learning rate by 0.97
        if i%200==0:
            print("Step {} Train accuracy: {:.4f}".format(i,np.mean(accuracies)))
            mode='eval'
            lr=lr*0.97
            accuracies=[]
            for k in range(np.shape(eval_xs1)[0]//batch_size):
                inputs=eval_input_function(eval_xs1,eval_xs2, eval_labels,eval_labels2, batch_size,k)
                accurs=sess.run(accuracy,feed_dict=
                        {x:inputs['x'],x2:inputs['x2'],y:inputs['y'],y2:inputs['y2']})
                for value in accurs:
                    if value==True:
                        accuracies.append(1)
                    else:
                        accuracies.append(0)
            mode='train'
            print("Eval accuracy: {:.4f}".format(np.mean(accuracies)))
            accuracies=[]
            savepath=saver.save(sess, model_dir)
            print('Model saved', savepath)
    
                

def train(model_dire):
  main(batch_size=25,mode='train',model_dir=model_dire)
 
def predict(pred_data1,pred_data2,predictions_file1,predictions_file2,model_dir):
    '''function for predicting with the network
    -pred_data1 is the path to first file containing spectrogram arrays for file to be predicted from
    -pred_data2 is the path to second input file from the same session,
    if none an array of zeroes will be fed as the second input
    -predictions_file1 and predictions_file2 are the paths to where predictions will be saved,
    -model_dir is the path to the saved model to be used for predicting'''
    fil1=open(pred_data1,'rb')
    predict_x1=np.load(fil1)
    fil1.close()
    if pred_data2!=None:
        fil2=open(pred_data2,'rb')
        predict_x2=np.load(fil2)
        fil2.close()
    else:
        predict_x2=np.zeros(np.shape(predict_x1),dtype=np.float16)
        
    predictions1,predictions2=main(predict_x1,predict_x2,batch_size=25,mode='predict',
                     model_dir=model_dir)
    
    np.save(predictions_file1,predictions1)
    np.save(predictions_file2,predictions2)


if __name__=='__main__':
  start=datetime.datetime.now()
  train("Models/model_small.ckpt")
  print("Time taken: ", datetime.datetime.now()-start)
  # predict('Data/20161219_Athos_Porthos_x1_test',
  #         'Data/20161219_Athos_Porthos_x2_test',
  #         'Data/pred_20161219_Athos_model_small',
  #         'Data/pred_20161219_Porthos_model_small',
  #         model_dir="Models/model_small.ckpt")
