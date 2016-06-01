
import os
import csv
import glob
import numpy as np
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav




(rate,sig) = wav.read("/Users/liudamabu/Desktop/MFCC/WAV/299.wav")
mfcc_feat = mfcc(sig,rate)

## read wav files
'''
c=os.listdir('/Users/liudamabu/Desktop/mfcc/WAV/')
for i in range(0, len(c)):
    (rate,sig) = wav.read(c[i])
    mfcc_feat = mfcc(sig,rate)
'''

'''

def get_features_vectors_wav(wav_file, features_on='all'):
    header = read_output_csv(wav_file)
    MFCC = []
    for index, row in header.iterrows():
        MFCC.append()
    MFCC = np.array(MFCC)
    return MFCC
'''
def _get_covariance_matrix(mfcc_feat):
    num_segments = len(mfcc_feat)
    num_features = len([0])
    
    cov_mat = []
    for i in range(num_features):
        tmp_mat = []
        for j in range(num_features):
            value = 0
            mean_1 = 0
            mean_2 = 0
            for k in range(num_segments):
                mean_1 += mfcc_feat[k][i]
                mean_2 += mfcc_feat[k][j]
                value += mfcc_feat[k][i] * mfcc_feat[k][j]
            mean_1 /= num_segments
            mean_2 /= num_segments
            value /= num_segments
            tmp_mat.append(value - mean_1 * mean_2)
        cov_mat.append(tmp_mat)
    return cov_mat

def _get_features_mean(mfcc_feat):
    num_segments = len(mfcc_feat)
    num_features = len(mfcc_feat[0])
    
    mean = [0 for i in range(num_features)]
    for i in range(num_segments):
        mean += mfcc_feat[i]
    for i in range(num_features):
        mean[i] /= num_segments
    
    return mean

def get_track_features(track_name):
    (rate,sig) = wav.read(track_name)
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    
    num_segments = len(mfcc_feat)
    num_features = len(mfcc_feat[0])
    
    features_mean = _get_features_mean(mfcc_feat)
    cov_mat = _get_covariance_matrix(mfcc_feat)
    
    return (features_mean, cov_mat)

'''
np.savetxt('/Users/liudamabu/Desktop/mfcc/mfcc1.txt',_get_covariance_matrix(mfcc_feat))
np.savetxt('/Users/liudamabu/Desktop/mfcc/mfcc2.txt',_get_features_mean(mfcc_feat))
'''

with open("/Users/liudamabu/Desktop/mfcc//MFCC/299_mfcc.csv", "w") as f:
    f_csv = csv.writer(f, dialect='excel',delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
    f_csv.writerow(_get_covariance_matrix(mfcc_feat))
    f_csv.writerow(_get_features_mean(mfcc_feat))
