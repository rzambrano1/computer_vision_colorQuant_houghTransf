#!/usr/bin/python3

# Standard Libraries
import argparse
import os
import sys
from tqdm import tqdm

# Type Hint Libraries
from typing import Optional, Tuple, Union, TypeVar, List
import numpy.typing as npt
import matplotlib.figure

# Math Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

# Machine Learning Libraries
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Image Libraries
import cv2 

import skimage as ski
from skimage import io
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb


def quantizeRGB(origImg: npt.NDArray[np.uint8],k: int, use_sampling: bool = True) -> Tuple[npt.NDArray[np.uint8],np.ndarray]:
    """
    Takes an RGB image as imput, quantizes the three dimensional RGB space, and maps each pixel in 
    the imput image to its nearest k-means center
    
    Input:
        origImg, an image with dimmensions MxNx3 of data type uint8
        k, an interger, specifies the number of colors to quantize to
        use_sampling, a boolean if True it fits the k-means model using a sub-sample of the image. Defatult to True
    
    Output:
        outputImg, an image with dimmensions MxNx3 of data type uint8 
        meanColors, a kx3 array of the k centers
    
    Parameters
    ----------
    origImg : np.ndarray [shape=(M,N,3)]
    k: int
    
    Returns
    -------
    outputImg: np.ndarray [shape=(M,N,3)]
    meanColors: np.ndarray[shape=(k,3)]
    
    Throws
    ------
    Raises:AssertionError, if the number of channels in the origImg parameter does not equal 3
    Raises:AssertionError, if the data type in the array of the origImg parameter is not np.uint8
    Raises:AssertionError, if the data type of the parameter k is not int
    Raises:AssertionError, if the parameter k is not greater than 1
    
    Examples
    --------
    >>>
    >>>
    """
    assert origImg.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert origImg.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    assert type(k) == int, 'k should be an integer.'
    assert k > 1, 'The number of k-means centers has to be greater than 1.'
    
    img = origImg.astype(np.double) # Converts the image's default 8 bits integer coding into double data type
    
    w, h, d = tuple(img.shape) # Records the images shape
    image_array = np.reshape(img, (w * h, d)) # Transforms the image into a 2D array
    
    # K-Means model
    model = KMeans(n_clusters=k,  random_state=42) 
    
    if use_sampling:
        
        # The idea of using sampling to was adaptep from scikit-learn user's guide, available at:
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
        
        image_array_sample = shuffle(image_array, random_state=42, n_samples=1_000)

        print('Fitting the model using sampling...')
        kmeans = model.fit(image_array_sample)
        labels = kmeans.predict(image_array)

        print('Model fit completed...')
        meanColors = model.cluster_centers_

        outputImg = meanColors[labels].reshape(w, h, d).astype(np.uint8)
        
    else:
        
        print('Fitting the model without sampling...')
        labels = model.fit_predict(image_array)

        print('Model fit completed...')
        meanColors = model.cluster_centers_

        outputImg = meanColors[labels].reshape(w, h, d).astype(np.uint8)
    
    return outputImg, meanColors