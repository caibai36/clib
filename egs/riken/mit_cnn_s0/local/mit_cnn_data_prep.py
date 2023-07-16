''' Creates a dataset(numpy array of spectrograms and corresponding numpy array of labels) given a list of wav files
and labels.'''
#necessary if running on a server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import signal
import random
import numpy as np
import os

#source directory for wav files and labels
directory="Wave_files/"
#result directory for saving the dataset
save_dir="Data/"

#list of 16bit wav file names to be used in creating the dataset
files=["20150814_Cricket_Enid/20150814_Cricket_aligned.wav","20150814_Cricket_Enid/20150814_Enid_aligned.wav",
       "20150903_Setta_Sailor/Setta_aligned.wav","20150903_Setta_Sailor/Sailor_aligned.wav"]

#list of label files associated with wav files, in the same order
label_files=["20150814_Cricket_Enid/Cricket.txt","20150814_Cricket_Enid/Enid.txt",
              "20150903_Setta_Sailor/Setta.txt","20150903_Setta_Sailor/Sailor.txt"]


#a dictionary for linking each call type to its nidex in label vector, symmetric to allow easy flipping
#when randomizing input order
truth_values={'cha':0,'chi':1,'ek':2,'ph':3,'ts':4,'tr':5,'trph':6,'tw':7,'noise':8,
              'tw2':9,'trph2':10,'tr2':11,'ts2':12,'ph2':13,'ek2':14,'chi2':15,'cha2':16}

#sessions to be used for evaluation
eval_sessions=[1]

train_data1=[]
train_data2=[]
correct=[]
eval_data1=[]
eval_data2=[]
eval_correct=[]

for k in range(int(len(files)/2)):
    lines1=[]
    current=0
    '''initialize first as a high number so we can keep track of where the first label is, saving the smallest
    value of current that has a label in order to only save data that has comes after the first label since the session before that
    has not been labeled and would end up in calls being marked as noise'''
    first=1000000
    #creates a list of labels out of text files with 150ms step size for discretization
    labels1=open(directory+label_files[2*k],'r')
    for line in labels1:
        start_t, end_t, typee=line.split('\t')
        '''there is no call in the middle 150ms of the window if the start of the next call is more than 325ms
        away from the start of the window. Keeps taking 150ms steps forward and adding noise to the list
        until the start of the call is in the middle'''
        while float(start_t)-current*0.15>0.325:
            current+=1
            lines1.append('noise')
        '''once there is a call in the middle 150ms of the window(ends more than 175ms from the start)
        adds its type to the list and steps 150ms forward'''
        while float(end_t)-current*0.15>0.175:
            if current<first:
                first=current
            lines1.append(typee[:-1].lower())
            current+=1
            
    lines2=[]
    current=0
    labels2=open(directory+label_files[2*k+1],'r')
    for line in labels2:
        start_t, end_t, typee=line.split('\t')
        while float(start_t)-current*0.15>0.325:
            current+=1
            lines2.append('noise')
        while float(end_t)-current*0.15>0.175:
            if current<first:
                first=current
            lines2.append(typee[:-1].lower())
            current+=1

    #lines is created to hold combination of lines1 and lines2
    lines=[]
    length=max(len(lines1),len(lines2))
    
    #pad the smaller list with noise to get to same size
    if len(lines1)<length:
        for i in range(length-len(lines1)):
            lines1.append('noise')
    elif len(lines2)<length:
        for i in range(length-len(lines2)):
            lines2.append('noise')

    '''creates a vector for each label initialized with 17 zeros. Changes the value at relevant index to one for each call type
    that is present and if no call present changes the value of noise to 1, call types not in the dictionary will be marked
    as noise'''      
    for i in range(length):
        init_y=np.zeros(17)
        if lines1[i]!='noise':
            try:
                init_y[truth_values[lines1[i]]]=1
            except(KeyError):
                pass
        if lines2[i]!='noise':
            try:
                init_y[truth_values[lines2[i]+'2']]=1
            except(KeyError):
                pass
        
        if np.sum(init_y)==0:
            init_y[truth_values['noise']]=1
        lines.append(init_y)
        
    nfft_a=512
    
    rate, data = wavfile.read(directory+files[2*k])
    rate2, data2 = wavfile.read(directory+files[2*k+1])

    #only looks between first and last label
    for i in range(first,length):
        #cretes the spectrogram if current piece is not labeled as noise and for for every fifth piece regardless
        if lines[i][8]!=1 or i%5==0:
            
            #take a 500ms piece out the data with step size of 150 ms
            signal_piece1=data[int(rate*i*0.15):int(rate*(i*0.15+0.5))]
            #resamples the signal piece if the file has a rate that is not 48kHz 
            if rate!=48000:
                signal_piece1=signal.resample(signal_piece1,int(len(signal_piece1)*48000/rate))

            #creates the spectrogram and scales it logarhitmically
            spectrum1,freqs1,time1,image1=plt.specgram(signal_piece1, NFFT=nfft_a, Fs=48000,
                                                       window=np.hamming(nfft_a), noverlap=420,
                                                       scale='linear', detrend='none')
            result1=10*np.log(abs(spectrum1+0.000001))
            #in case the shape is not what it should be skip this piece
            if np.shape(result1)!=(257,256):
                print("invalid shape1", np.shape(result1))
                continue
            
            #the same for second wav file
            signal_piece2=data2[int(i*rate2*0.15):int(rate2*((i+1)*0.15+0.35))]
            if rate2!=48000:
                signal_piece2=signal.resample(signal_piece2,int(len(signal_piece2)*48000/rate2))
                
            spectrum2,freqs2,time2,image2=plt.specgram(signal_piece2, NFFT=nfft_a, Fs=48000,
                                                       window=np.hamming(nfft_a), noverlap=420,
                                                       scale='linear', detrend='none')
            result2=10*np.log(abs(spectrum2+0.000001))
            if np.shape(result2)!=(257,256):
                print("invalid shape2", np.shape(result2))
                continue

            #adds arrays to the correct set
            if k in eval_sessions:
                eval_correct.append(lines[i])
                eval_data1.append(np.array(result1,dtype=np.float16))
                eval_data2.append(np.array(result2,dtype=np.float16))
            else:
                correct.append(lines[i])
                train_data1.append(np.array(result1,dtype=np.float16))
                train_data2.append(np.array(result2,dtype=np.float16))
            plt.clf()
            if i%1000==0:
                print(k,i)


print('Length of dataset:', len(train_data1))

train_data1=np.array(train_data1,dtype=np.float16)
train_data2=np.array(train_data2,dtype=np.float16)
correct=np.array(correct,dtype=np.float16)
eval_data1=np.array(eval_data1,dtype=np.float16)
eval_data2=np.array(eval_data2,dtype=np.float16)
eval_correct=np.array(eval_correct,dtype=np.float16)


if not os.path.exists(save_dir):
    os.mkdir(save_dir)
                               
#train specgrams for animal1
save_fil1=open(save_dir+'train_xs1','wb')
np.save(save_fil1,train_data1)
save_fil1.close()
#train specgrams for animal2
save_fil2=open(save_dir+'train_xs2','wb')
np.save(save_fil2,train_data2)
save_fil2.close()
#train labels
save_fil3=open(save_dir+'train_ys_multi','wb')
np.save(save_fil3,correct)
save_fil3.close()

#evaluation pecgrams for animal1
save_fil4=open(save_dir+'eval_xs1','wb')
np.save(save_fil4,eval_data1)
save_fil4.close()
#evaluation pecgrams for animal2
save_fil5=open(save_dir+'eval_xs2','wb')
np.save(save_fil5,eval_data2)
save_fil5.close()
#evaluation labels
save_fil6=open(save_dir+'eval_ys_multi','wb')
np.save(save_fil6,eval_correct)
save_fil6.close()
print('Saved')

