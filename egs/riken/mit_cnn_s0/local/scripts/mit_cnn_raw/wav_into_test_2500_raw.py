'''
Creates spectrogram arrays out of wav files to be used for testing. Makes a spectrogram for 2500ms piece
at a time to improve performance
'''
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy
import os


def create_spec_data(wav_file1,save_loc1):
    x1=[]
    rate, data = wavfile.read(wav_file1)
    length=len(data)
    for i in range(int(length/(rate*2.5)-1)):
        signal_piece=data[int(rate*2.5*i):int(rate*(2.5*(i+1)))]
        if rate!=48000:
            signal_piece=signal.resample(signal_piece,int(len(signal_piece)*48000/rate))
        spectrum,freqs,time,image=plt.specgram(signal_piece, NFFT=512, Fs=48000,
                                               window=numpy.hamming(512), noverlap=420,
                                               scale='linear', detrend='none')
        spectrum=10*numpy.log(abs(spectrum+0.000001))
        b=numpy.array(spectrum,dtype=numpy.float16)
        if numpy.shape(b)!=(257,1299):
            print('Invalid array shape: ',numpy.shape(b))
            continue
        x1.append(b)
        if i%100==0:
            print("{}/{}".format(i,length//(rate*2.5)))
        plt.clf() 
             
    save_fil1=open(save_loc1,'wb')
    numpy.save(save_fil1,numpy.array(x1,dtype=numpy.float16))
    save_fil1.close()

if __name__=="__main__":
    create_spec_data("Wave_files/20161219_Athos_Porthos/Athos_20161219_aligned.wav",
                     "Data/20161219_Athos_Porthos_x1_test")
    create_spec_data("Wave_files/20161219_Athos_Porthos/Porthos_20161219_aligned.wav",
                     "Data/20161219_Athos_Porthos_x2_test")
    
                     
