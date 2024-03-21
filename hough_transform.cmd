:: This CMD script runs the main function for the detecting circles with Hough Transform script
:: The goal is to detect circles of a fixed rarius in an image
:: As per instructions in the assignment the images eyes_deer.jpg and sports_balls.jpg have been used. 

::=====================
::Programming Problem 1
::=====================
@echo off

TITLE Detecting Circles with Hough Transform

ECHO Starting script for eyes_deer.jpg image... 

python houghTransformMain.py -in eyes_deer -r 2 -c 10 -b 5

ECHO Starting script for sports_balls.jpg image... 

python houghTransformMain.py -in sports_balls -r 25 -c 2 -b 5

python houghTransformMain.py -in sports_balls -r 75 -c 2 -b 5