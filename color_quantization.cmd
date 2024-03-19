:: This CMD script runs the main function for the color quantization script
:: The goal is to use both the RGB space and the Hue space from the HSV format to quantize a given image.
:: As per instructions in the assignment the image baloons.jpg has been used. 

::=====================
::Programming Problem 1
::=====================
@echo off

TITLE Colour Quantizing

ECHO Starting script... 

python colorQuantizemain.py -in baloons --low_k 4 --high_k 16