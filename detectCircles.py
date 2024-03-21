#!/usr/bin/python3

# Standard Libraries
import argparse
import os
import sys
import time
from tqdm import tqdm
from collections import Counter
import re 

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
from skimage.filters import threshold_otsu
from skimage.feature import canny


def get_coord(accumulator_elem):
    """
    This is a helper function that converts the string that contains the coordinates into a numpy array.
    Given the accumulator was initialized as a Counter() object, a mapping, the keys must be strings. This makes
    easy count the votes, but comes with the caveat of having to convert the keys of the counter back to an
    array of integers.
    """
    return np.array([int(coord) for coord in re.sub(r'[^\d|\.]+', '', accumulator_elem[0])[:-1].split('.')])

def detectCircles(im: npt.NDArray[np.uint8], radius: float, useGradient: bool=False)-> npt.NDArray:
    """
    Implementation of Hough Transform to detect circles about the radius size of of the parameter 'radius'
    
    Input:
        im, an image with dimmensions MxNx3 of data type uint8
        radius, a float type, specifies the size of circle the function will be looking for
        useGradient, a boolen type, is a flag that allows the function's client to optionally exploit the gradient direction
        measured at the edge points. Defaults to False
    
    Output:
        centers,  an kx2 numpy array, in which each row lists the (x, y) position of a detected circles' center
        
    Parameters
    ----------
    im: np.ndarray [shape=(M,N,3)]
    radius: float
    useGradient: bool
    
    Returns
    -------
    centers: np.ndarray [shape=(k,2)]
    
    Throws
    ------
    Raises:AssertionError, if the number of channels in 'im' parameter does not equal 3
    Raises:AssertionError, if the data type in the array of 'im' parameter is not np.uint8
    Raises:AssertionError, if the useGradient flag is not a boolean type
    Raises:ValueError, if radius is not of type float and the provided parameter's type cannot be casted into float
    Raises:AssertionError, if value of radius is equal to zero or a negative number
    
    Examples
    --------
    >>>
    >>>
    """
    assert im.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert im.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    assert type(useGradient)==bool, 'useGradient flag must be a boolen type.'
    
    if type(radius) != float:
        print("Casting radius into float type...")
        try:
            radius = float(radius)
        except:
            print("Unexpected data type on the radius parameter. Please provide a valid data type, in this case float...")
            print("... Cannot run detectCircles(). Try again!")
            return -1
    
    assert radius > 0, 'The radius parameter must be greater than 0.'
    
    # Transform the RGB image into grayscale image
    im_gray = ski.color.rgb2gray(im)
    
    # Apply Canny edge detection to the gray image
    im_edges = canny(im_gray, sigma=.70,low_threshold=0.55, high_threshold=0.8)
    
    # Finding coordinates for edges
    edges = np.argwhere(im_edges>0) 
    
    if useGradient:
        
        ### Creating Partial Derivative Operators ###
    
        # 2D version of the Scharr operator
        filter_dy = np.array([
            [47.0, 162.0, 47.0],
            [0.0, 0.0, 0.0],
            [-47.0, -162.0, -47.0],
        ])

        # 2D version of the Scharr operator
        filter_dx = np.array([
            [47.0, 0.0, -47.0],
            [162.0, 0.0, -162.0],
            [47.0, 0.0, -47.0],
        ])
        
        accumulator = Counter() # Initializes the accumulator
        h,w = im_gray.shape # Recording the height and width of the image to handle border points
        
        for row,col in edges:
            
            if (row>0 and col>0) and (row<h and col<w):
                
                # Extracts the pixel values in the gray image on the neighborhood of the pixed in the edge
                neighborhood = np.array([
                    [im_gray[row-1,col-1],im_gray[row-1,col],im_gray[row-1,col+1]],
                    [im_gray[row,col-1],im_gray[row,col],im_gray[row,col+1]],
                    [im_gray[row+1,col-1],im_gray[row+1,col],im_gray[row+1,col+1]]
                ])

                # Computes estimated partial derivatives in x-direction and y-direction
                dx = np.sum(np.multiply(neighborhood, filter_dx))
                dy = np.sum(np.multiply(neighborhood, filter_dy))

                # Computes local direction of the gradient at the pixel lying on the edge
                # https://en.wikipedia.org/wiki/Image_gradient
                theta = np.arctan(dy/dy)
                
                # Computes the magnitude of the gradient
                # Using the magnitude as a weight to provide more votes to stronger edges
                # After testing it improves the output of the algorithm
                magnitude = np.sqrt(dy*dy+dx*dx)

                a = np.array([int(row - np.round(radius*np.cos(theta)))])
                b = np.array([int(col + np.round(radius*np.sin(theta)))])
                circle_parameters = np.c_[float(a),float(b)][0] # No need to trace all points in the circle located in the
                                               # Hough space for x,y edge point since we estimated the likely parameters
                                               # using the diection of the gradient at this point x,y
                accumulator[str(circle_parameters)] += 1*magnitude
                
            else: # Adding this line for readability. If the edge lies at the borther of the image, just skip
                pass
        
    else:
        
        angles = np.linspace(0, 2*np.pi, 360) # Pre-computing angles to draw the circles in Hough space
        accumulator = Counter() # Initializes the accumulator

        for x,y in edges:
            a = np.array([])
            b = np.array([])
            for theta in angles:
                aa = int(x - np.round(radius*np.cos(theta)))
                a = np.append(a,[aa])
                bb = int(y + np.round(radius*np.sin(theta)))
                b = np.append(b,[bb])
            circle = np.c_[a,b] # circle in Hough space for x,y edge point
            for coord in circle:
                accumulator[str(coord)] += 1
    
    total_centers_houghSpace = len(accumulator)
    # top_ten_perc = int(total_centers_houghSpace*0.1) # This line of code is ment to save space, but will not be used
    centers = np.zeros(shape=(total_centers_houghSpace,3),dtype=int)
    
    for i,elem in enumerate(accumulator.most_common()): # Get elements in descending order by using most_common() method 
        y,x = get_coord(elem)
        centers[i,:] = x,y,elem[1]
        
    return centers