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

# Functions Import
from quantizeHSV import quantizeHSV


def getHueHists(im: npt.NDArray[np.uint8], k: int)-> Tuple[matplotlib.figure.Figure,matplotlib.figure.Figure]:
    """
    Computes and display two histograms of the hue values of the input image. The first histogram uses 
    equally-spaced bins (uniformly dividing up the hue values). The second histogram use bins defined 
    by the k cluster center memberships.
    
    Input:
        im, a matrix with dimmensions MxNx3 of data type uint8 that represents an image
        k, an interger, specifies the number of colors to quantize to
    
    Output:
        histEqual, a histogram that uses equally-spaced bins (uniformly dividing up the hue values)
        histclustered, a  histogram that uses bins defined by the k cluster center memberships 
        (ie., all pixels belonging to hue cluster i go to the i-th bin, for i=1,...k)
    
    Parameters
    ----------
    im : np.ndarray [shape=(M,N,3)]
    k: int
    
    Returns
    -------
    histEqual: matplotlib.figure
    histclustered: matplotlib.figure
    
    Throws
    ------
    Raises:AssertionError, if the number of channels in im parameter does not equal 3
    Raises:AssertionError, if the data type in the array of the im parameter is not np.uint8
    Raises:ValueError, if k is not of type int and the provided parameter's type cannot be casted into int 
    
    Examples
    --------
    >>>
    >>>
    """
    assert im.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert im.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    if type(k) != int:
        print("Converting parameter k into integer type...")
        try:
            k = int(k)
        except:
            print("Unexpected data type on parameter k. Please provide a valid data type, in this case integer...")
            print("... Cannot run getHueHists(). Try again!")
            return -1
    
    hsv_img = rgb2hsv(im) # Converts he original image from RGB to HSV
    hue_img = hsv_img[:, :, 0] # Extracts the hue layer from the original image
    
    quantized_im, k_centers = quantizeHSV(origImg = im, k = k)
    
    histEqual = plt.figure()
    plt.hist(hue_img.ravel(), 36)
    plt.title("Histogram of the Hue channel")
    
    hsv_q_img = rgb2hsv(quantized_im)
    hue_q_img = hsv_q_img[:, :, 0]
    
    histclustered = plt.figure()
    plt.hist(hue_q_img.ravel(), 36)
    plt.title("Histogram of the Hue channel of Quantized Image")
    
    return histEqual, histclustered