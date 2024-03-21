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

# Import Functions
from detectCircles import get_coord
from detectCircles import detectCircles

# Helper Functions

def circle_points(x_coord_center,y_coord_center,radius):
    """
    Helper function to draw a circle from a given center x,y and a given radius.
    """
    angles = np.linspace(0, 2*np.pi, 360*4) # Pre computing angles to draw the circles in Hough space
    x = np.array([])
    y = np.array([])
    for angle in angles:
        xx = int(x_coord_center - np.round(radius*np.cos(angle)))
        #xx = int(np.round(radius*np.cos(angle)))
        x = np.append(x,[xx])
        yy = int(y_coord_center + np.round(radius*np.sin(angle)))
        #yy = int(np.round(radius*np.sin(angle)))
        y = np.append(y,[yy])
    circle = np.c_[x,y]
    
    return circle

def post_process_centers(centers_matrix,im_gray,bin_size_2d):
    """
    Helper function to aggregate votes in Hough Transform in centers hat are close by.
    """
    
    h,w = im_gray.shape

    bin_size = bin_size_2d
    h_ = np.arange(bin_size,h+bin_size,bin_size)
    w_ = np.arange(bin_size,w+bin_size,bin_size)
    ww, hh = np.meshgrid(w_, h_)

    count = np.zeros_like(hh)

    for center in centers_matrix:
        pixel_pos_w_dir = center[0] 
        pixel_pos_h_dir = center[1]
        votes = center[2]
        
        try:
            bin_index_w_dir = np.where(ww[0,:] > pixel_pos_w_dir)[0][0]
        except:
            bin_index_w_dir = np.where(ww[0,:] > pixel_pos_w_dir)[0]

        try:
            bin_index_h_dir = np.where(hh[:,0] > pixel_pos_h_dir)[0][0]
        except:
            bin_index_h_dir = np.where(hh[:,0] > pixel_pos_h_dir)[0]

        count[bin_index_h_dir,bin_index_w_dir] = count[bin_index_h_dir,bin_index_w_dir] + votes

    aggregated_accumulator = Counter()

    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            total_votes = count[i,j]
            if total_votes > 0:
                center_in_w = (ww[i,j]-bin_size+ww[i,j])/2
                center_in_h = (hh[i,j]-bin_size+hh[i,j])/2
                coord = np.array([float(int(center_in_w)),float(int(center_in_h))]) # Using float(int()) to round and then add the dot in float, in order to meet forat expected by regular expression
                aggregated_accumulator[str(coord)] += total_votes

    aggregated_centers = np.zeros(shape=(len(aggregated_accumulator),3),dtype=int)

    for i,elem in enumerate(aggregated_accumulator.most_common()): # Get elements in descending order by using most_common() method 
        x,y = get_coord(elem)
        aggregated_centers[i,:] = x,y,elem[1]

    return aggregated_centers

def show_circles(im_2D,center_matrix,num_circles,r,title, col='green'):
    """
    Helper function to show detected circles in image
    """
    
    plt.imshow(im_2D, cmap='gray')
    
    for i in range(num_circles):
        circle = circle_points(center_matrix[i,0],center_matrix[i,1],r)
        plt.plot(circle[:,0], circle[:,1], color=col)
    
    title_string = title + '' + str(r)
    plt.title(title_string)
    plt.show()
    
    return 0

# Main Function

def main(input_file_name: str, radius: int, num_circles: int, bin_size: int):
    
    if input_file_name[-4:] == '.jpg':
        PATH_IMG = '..\\Zambrano_Ricardo_ASN3_py\\' + input_file_name
        out_name = input_file_name[:-4]
    else:
        PATH_IMG = '..\\Zambrano_Ricardo_ASN3_py\\' + input_file_name + '.jpg'
        out_name = input_file_name

    print("Loading image...")
    img_raw = io.imread(PATH_IMG)
    img = img_raw.copy()
    
    # Preprocessing images for display later
    img_gray = ski.color.rgb2gray(img)
    img_edges = canny(img_gray, sigma=.70,low_threshold=0.55, high_threshold=0.8)

    print('Running Hough Transform for Circles - No Gradient...')
    start = time.time()
    test1 = detectCircles(img, radius=radius, useGradient=False)
    end = time.time()
    print('Runtime for detecting circles with Hough Transform without exploiting gradient: ',end - start)
    print(test1)

    # Showing images with standard accumulator output
    show_circles(img_edges,test1,num_circles,radius,'Hough Transform for Circles - No Gradient - Radius = ', col='green')
    show_circles(img_gray,test1,num_circles,radius,'Hough Transform for Circles - No Gradient - Radius = ', col='green')

    print('Running Hough Transform for Circles - Exploiting Gradient...')
    start = time.time()
    test2 = detectCircles(img, radius=radius, useGradient=True)
    end = time.time()
    print('Runtime for detecting circles with Hough Transform exploiting gradient: ',end - start)
    print(test2)

    # Showing images with standard accumulator output
    show_circles(img_edges,test2,num_circles,radius,'Hough Transform for Circles - Exploiting Gradient - Radius = ', col='red')
    show_circles(img_gray,test2,num_circles,radius,'Hough Transform for Circles - Exploiting Gradient - Radius = ', col='red')

    # Post processing accumulator
    new_centers1 = post_process_centers(test1,img_gray,bin_size) # New centers for Hough transform without Gradient
    new_centers2 = post_process_centers(test2,img_gray,bin_size) # New centers for Hough transform without Gradient

    # Showing images with AGGREGATED accumulator output
    show_circles(img_edges,new_centers1,num_circles,radius,'Hough Transform - Post-Processed Accumulator - No Gradient - Radius = ', col='blue')
    show_circles(img_gray,new_centers1,num_circles,radius,'Hough Transform - Post-Processed Accumulator - No Gradient - Radius = ', col='blue')

    show_circles(img_edges,new_centers2,num_circles,radius,'Hough Transform - Post-Processed Accumulator - Exploiting Gradient - Radius = ', col='cyan')
    show_circles(img_gray,new_centers2,num_circles,radius,'Hough Transform - Post-Processed Accumulator - Exploiting Gradient - Radius = ', col='cyan')

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pass a string with the name JPG file in the folder and a radius of circles that the algorithm will searc hfor in the image')

    parser.add_argument('-in','--input_file_name', type=str, default=False, action='store', required=True, help="String with name of JPG file in folder")
    parser.add_argument('-r','--radius', type=int, default=2, action='store', required=True, help="Radius of circles to detect in the image")
    parser.add_argument('-c','--num_circles', type=int, default=10, action='store', required=True, help="Provide number of circles to display in the image")
    parser.add_argument('-b','--bin_size', type=int, default=5, action='store', required=True, help="Provide the size of bins for aggregating detected centers that are close to each other")
    
    args = parser.parse_args()
    main(str(args.input_file_name), int(args.radius), int(args.num_circles), int(args.bin_size))
