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


def quantizeHSV(origImg: npt.NDArray[np.uint8],k: int, use_sampling: bool = True) -> Tuple[npt.NDArray[np.uint8],np.ndarray]:
    """
    Takes an RGB image as imput, converts it to HSV and then quantizes the one dimensional hue space, 
    Next it maps each pixel in the input image to its nearest quantized hue value while keeping its saturation
    and value channelsthe same as the input. Finally it converts the image back to RGB
    
    Input:
        origImg, an image with dimmensions MxNx3 of data type uint8
        k, an interger, specifies the number of colors to quantize to
        use_sampling, a boolean if True it fits the k-means model using a sub-sample of the image. Defatult to True
    
    Output:
        outputImg, an image with dimmensions MxNx3 of data type uint8 
        meanHues, a kx1 array of the k hue centers
    
    Parameters
    ----------
    origImg : np.ndarray [shape=(M,N,3)]
    k: int
    
    Returns
    -------
    outputImg: np.ndarray [shape=(M,N,3)]
    meanHues: np.ndarray[shape=(k,1)]
    
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
    
    hsv_img = rgb2hsv(origImg) # Converts he original image from RGB to HSV
    
    w, h, _ = tuple(hsv_img.shape) # Records the images shape
    d = 1 # The image has 3 channels, however, the k-mean centers will be calculated only in the 1-d hue channel. This is why d=1
    
    # Separate the channels to isolate them
    hue_img = hsv_img[:, :, 0]
    saturation_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    
    image_array = np.reshape(hue_img, (w * h, d)) # Reshapes the image to be processed by k-means function
    
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
        meanHues = model.cluster_centers_

        outputHue = meanHues[labels].reshape(w, h)
        
        outputImg = np.stack((outputHue, saturation_img, value_img), axis=2)
        outputImg = hsv2rgb(outputImg)
        outputImg = outputImg*255
        outputImg = outputImg.astype(np.uint8)
        
    else:
        
        print('Fitting the model without sampling...')
        labels = model.fit_predict(image_array)

        print('Model fit completed...')
        meanHues = model.cluster_centers_

        outputHue = meanHues[labels].reshape(w, h)   
        
        outputImg = np.stack((outputHue, saturation_img, value_img), axis=2)
        outputImg = hsv2rgb(outputImg)
        outputImg = outputImg*255
        outputImg = outputImg.astype(np.uint8)
    
    return outputImg, meanHues