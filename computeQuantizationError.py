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


def computeQuantizationError(origImg: npt.NDArray[np.uint8],quantizedImg: npt.NDArray[np.uint8])-> float:
    """
    Computes the sum of squared error (SSD error) between the pixel values of an RGB image and the pixel values of a 
    quantized version of the image.
    
    Input:
        origImg, an image with dimmensions MxNx3 of data type uint8
        quantizedImg, an image with dimmensions MxNx3 of data type uint8. It is assumed this image is a quantized version
        of origImg
    
    Output:
        error, sum of squared error, a float
        
    Parameters
    ----------
    origImg : np.ndarray [shape=(M,N,3)]
    quantizedImg: np.ndarray [shape=(M,N,3)]
    
    Returns
    -------
    error: float
    
    Throws
    ------
    Raises:AssertionError, if the number of channels in the origImg or quantizedImg parameters do not equal 3
    Raises:AssertionError, if the data type in the array of the origImg or quantizedImg parameters are not np.uint8
    Raises:AssertionError, if the images dimensions do not match
    
    Examples
    --------
    >>>
    >>>
    """
    assert origImg.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert origImg.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    assert quantizedImg.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert quantizedImg.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'    
    
    w, h, d = tuple(origImg.shape) # Records the images shape
    x, y, z = tuple(quantizedImg.shape) # Records the images shape
    
    assert (w==x) and (h==y) and (d==z), 'The images must have the same size.'
    
    img = origImg.astype(np.double) # Converts the image's default 8 bits integer coding into double data type
    qimg = quantizedImg.astype(np.double) # Converts the image's default 8 bits integer coding into double data type
    
    error = np.sum((img - qimg) ** 2)
    
    return error