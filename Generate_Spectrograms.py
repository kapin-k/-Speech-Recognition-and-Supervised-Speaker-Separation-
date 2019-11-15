import os

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib import cm
import pylab

import librosa
from librosa import display
import numpy as np


for root, dirs, files in os.walk("./smallcorpus_VoxCeleb", topdown=False):
       if root == "./smallcorpus_VoxCeleb":
        for name in dirs:
            #X =y = os.listdir(root+"/"+name)
            #wav_file=os.listdir(root+"/"+name)
            WAV_DIR="./smallcorpus_VoxCeleb/"+name+"/"
            wav_file=os.listdir(WAV_DIR)
            IMG_DIR="./spectrogram_images/"+name+"/"
            for f in (wav_file):
                try:
       		         # Read wav-file
                   y, sr = librosa.load(WAV_DIR+f)
      			   # Use the default sampling rate of 22,050 Hz
                   print(sr);   
     			   # Pre-emphasis filter
                   pre_emphasis = 0.97
                   y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        	         # Compute spectrogram
                   M = librosa.feature.melspectrogram(y, sr, 
                                      				fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
                                       				n_fft=2048, 
                                       			        hop_length=512, # window hop
                                       				n_mels = 96, # As per the Google Large-scale audio CNN paper
                                       				power = 2) # Power = 2 refers to squared amplitude
        
        		  # Power in DB
                   log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
                         # Plotting the spectrogram and save as JPG without axes (just the image)
                   pylab.figure(figsize=(4,4))
                   pylab.axis('off') 
                   pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
                   librosa.display.specshow(log_power, cmap=cm.jet)
                   pylab.savefig(IMG_DIR + f[:-4]+'.jpg', bbox_inches=None, pad_inches=0)
                   pylab.close()
		
                except Exception as e:
                       print(f, e)
                       pass
