'''
Creates spectrogram arrays out of wav files to be used for testing. Makes a spectrogram for 2500ms piece
at a time to improve performance
'''
#necessary if running on a server
import matplotlib as mpl
mpl.use('Agg')
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

# if __name__=="__main__":
#     create_spec_data("Wave_files/20161219_Athos_Porthos/Athos_20161219_aligned.wav",
#                      "Data/20161219_Athos_Porthos_x1_test")
#     create_spec_data("Wave_files/20161219_Athos_Porthos/Porthos_20161219_aligned.wav",
#                      "Data/20161219_Athos_Porthos_x2_test")

import argparse
import json
parser = argparse.ArgumentParser(description="Create 2500ms spectral segments to prepare the inputs of test sets. Split audio pairs into 2500ms segments and take the spectral frames for each segment (reference: 'wav_into_test_2500_raw.py' from https://marmosetbehavior.mit.edu/).")
parser.add_argument("--info_json", type=str, default="data/mit_sample/info.json", help="json file that maps utterance id to key-value pairs. The keys should include 'wav' and 'aud' for locations of audio files and audacity labels. {'uttid1': {'wav1': wav1_path, 'aud1': audacity_label1_path}, 'uttid2': {'wav2': wav2_path, 'aud2': audacity_label2_path}}. Each line of audacity label file is 'begin_time_sec end_time_sec label'.")
parser.add_argument("--test_wav_uttids", type=str, default=["Athos", "Porthos"], nargs="+", help='uttid pairs for test sets. e.g., "--test_wav_uttids Cricket Enid Setta Sailor" where "Crick" and "Enid" are the first pair; "Setta" and "Sailor" are the second pair.')
parser.add_argument("--test_wav_paths", type=str, default=[], nargs="+", help='path pairs for test sets. Instead of passing a sequence of uttids by --test_wav_uttids, this option directly passes paths of wav files. The info_json file is not needed here.')
parser.add_argument("--out_dir", type=str, default="exp/data/mit_sample", help="Output directory to store the spectra of 2500ms segments to prepare the inputs of test sets. e.g., the output file would be test_input1_uttid1 and test_input2_wavfilename (if wav paths are provided).")
args = parser.parse_args()

print(args)

with open(args.info_json) as f:
    info = json.load(f)

if args.test_wav_uttids: wavs = [info[uttid]['wav'] for uttid in args.test_wav_uttids]
if args.test_wav_paths: wavs = args.test_wav_path

save_dir = args.out_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for k in range(int(len(wavs)/2)):
    #     create_spec_data("Wave_files/20161219_Athos_Porthos/Athos_20161219_aligned.wav",
    #                      "Data/20161219_Athos_Porthos_x1_test")
    #     create_spec_data("Wave_files/20161219_Athos_Porthos/Porthos_20161219_aligned.wav",
    #                      "Data/20161219_Athos_Porthos_x2_test")
    if args.test_wav_uttids: name = args.test_wav_uttids[2*k]
    if args.test_wav_paths: name = os.path.splitext(os.path.basename(wavs[2*k]))[0] # "pair1/pair1_animal1.wav" => "pair1_animal1"
    create_spec_data(wavs[2*k], os.path.join(args.out_dir, "test_input1_" + name))

    if args.test_wav_uttids: name = args.test_wav_uttids[2*k+1]
    if args.test_wav_paths: name = os.path.splitext(os.path.basename(wavs[2*k+1]))[0] # "pair1/pair1_animal1.wav" => "pair1_animal1"
    create_spec_data(wavs[2*k+1], os.path.join(args.out_dir, "test_input2_" + name))
