import os

import glob

import pandas as pd
import numpy as np
import mlpy.wavelet as wavelet

import cv2

from sklearn import preprocessing
from sklearn.decomposition import PCA

from tabulate import tabulate

import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves


pd.set_option('display.multi_sparse', False)

classes = ["brush",
           "openbinder",
           "closebinder",
           "erase",
           "flippage",
           "plugheadphones",
           "unplugheadphone",
           "putheadphones",
           "removeheadphones",
           "rip",
           "type",
           "write"
           ]

for fileName in glob.glob('./output/*'):
    thisClass = [c for c in classes if os.path.basename(fileName).lower().find(c)==0][0]
    store = pd.HDFStore(fileName)

    timestep = store['timestep']

    estimated = store['estimated']
    observed = store['observed']

    estimated_pca = store['estimated_pca']
    observed_pca = store['observed_pca']


    print fileName, thisClass
    print "Original:", estimated.values.shape, observed.values.shape
    print "PCA:", estimated_pca.values.shape, observed_pca.values.shape

    print "-"*40
    store.close()
