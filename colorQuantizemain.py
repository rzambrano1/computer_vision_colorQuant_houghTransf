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
from quantizeRGB import quantizeRGB
from getHueHists import getHueHists
from computeQuantizationError import computeQuantizationError


def main(input_file_name, low_k, high_k):

    if input_file_name[-4:] == '.jpg':
        PATH_IMG = '..\\Zambrano_Ricardo_ASN3_py\\' + input_file_name
        out_name = input_file_name[:-4]
    else:
        PATH_IMG = '..\\Zambrano_Ricardo_ASN3_py\\' + input_file_name + '.jpg'
        out_name = input_file_name
    
    PATH_RGB_LOW_K = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_rgb_quantized_low_k.png'
    PATH_RGB_HIGH_K = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_rgb_quantized_high_k.png'
    PATH_HSV_LOW_K = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_hsv_quantized_low_k.png'
    PATH_HSV_HIGH_K = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_hsv_quantized_high_k.png'
    PATH_HIST_OUT_1 = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_image_histogram.png'
    PATH_HIST_OUT_2 = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_quantized_lowK_histogram.png'
    PATH_HIST_OUT_3 = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_quantized_highK_histogram.png'
    PATH_FINAL_OUT = '..\\Zambrano_Ricardo_ASN3_py\\' + out_name + '_histogram.png'

    print("Loading image...")
    img_raw = io.imread(PATH_IMG)
    img = img_raw.copy()

    # Showing original image 
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()

    print("RGB space quantizing...")
    quantized_rgb_low, rgb_colCenters_low = quantizeRGB(img,low_k)

    # Showing quantized image 
    plt.imshow(quantized_rgb_low)
    plt.title("Image Quantized Using the RGB Space - Low K")
    plt.show()

    quantized_rgb_high, rgb_colCenters_high = quantizeRGB(img,high_k)

    # Showing quantized image 
    plt.imshow(quantized_rgb_high)
    plt.title("Image Quantized Using the RGB Space - High K")
    plt.show()

    io.imsave(PATH_RGB_LOW_K,quantized_rgb_low)
    io.imsave(PATH_RGB_HIGH_K,quantized_rgb_high)

    print("HSV space quantizing...")
    quantized_hsv_low, hsv_hueCenters_low = quantizeHSV(img,low_k)

    # Showing quantized image 
    plt.imshow(quantized_hsv_low)
    plt.title("Image Quantized Using the Hue Space - Low K")
    plt.show()

    quantized_hsv_high, hsv_hueCenters_high = quantizeHSV(img,high_k)

    # Showing quantized image 
    plt.imshow(quantized_hsv_high)
    plt.title("Image Quantized Using the Hue Space - High K")
    plt.show()

    io.imsave(PATH_HSV_LOW_K,quantized_hsv_low)
    io.imsave(PATH_HSV_HIGH_K,quantized_hsv_high)

    print("Calculating SSD error of the quantized images...")

    error_rgb_k_low = computeQuantizationError(img,quantized_rgb_low)
    error_rgb_k_high = computeQuantizationError(img,quantized_rgb_high)
    error_hsv_k_low = computeQuantizationError(img,quantized_hsv_low)
    error_hsv_k_high = computeQuantizationError(img,quantized_hsv_high)

    print("The SSD Error attained when quantizing in the RGB space with low k was equal to: ", error_rgb_k_low)
    print("The SSD Error attained when quantizing in the RGB space with high k was equal to: ", error_rgb_k_high)
    print("The SSD Error attained when quantizing in the Hue space with low k was equal to: ", error_hsv_k_low)
    print("The SSD Error attained when quantizing in the Hue space with high k was equal to: ", error_hsv_k_high)

    print("\n---------------------------------------------------------------------------------\n")
    print("Calculating histograms...")

    histEqual, histclustered_low_k = getHueHists(img, low_k)
    _, histclustered_high_k = getHueHists(img, high_k)

    # Histogram Output

    histEqual.savefig(PATH_HIST_OUT_1, format='png')
    histclustered_low_k.savefig(PATH_HIST_OUT_2, format='png')
    histclustered_high_k.savefig(PATH_HIST_OUT_3, format='png')

    hist1 = io.imread(PATH_HIST_OUT_1)
    hist2 = io.imread(PATH_HIST_OUT_2)
    hist3 = io.imread(PATH_HIST_OUT_3)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    ax = axes.ravel()

    ax[0].imshow(hist1)
    #ax[0].set_title("Original Image")

    ax[1].imshow(hist2)
    ax[1].set_title("Low K")

    ax[2].imshow(hist3)
    ax[2].set_title("High K")

    fig.tight_layout()

    plt.savefig(PATH_FINAL_OUT)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pass a string with the name JPG file in the folder, a low value for the number of clusters in the k-means algorithm, and a high value for the number of clusters in the k-means algorithm')

    parser.add_argument('-in','--input_file_name', type=str, default=False, action='store', required=True, help="String with name of JPG file in folder")
    parser.add_argument('-low','--low_k', type=int, default=4, action='store', required=True, help="Integer indicating low number of clusters")
    parser.add_argument('-high','--high_k', type=int, default=16, action='store', required=True, help="Integer indicating high number of clusters")
    
    args = parser.parse_args()
    main(str(args.input_file_name), int(args.low_k), int(args.high_k))

