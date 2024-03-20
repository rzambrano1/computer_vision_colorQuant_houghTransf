#!/usr/bin/python3

# Standard Libraries
import argparse
import os
import sys
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


