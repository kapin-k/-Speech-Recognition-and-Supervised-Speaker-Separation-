import os

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib import cm
import pylab

import librosa
from librosa import display
import numpy as np

import pickle
import numpy as np
import librosa
from scipy.io.wavfile import read
from sklearn.mixture import GMM 
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy
from sklearn import metrics

from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import os
import sys
import numpy as np
