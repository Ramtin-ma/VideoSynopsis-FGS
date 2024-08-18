# *************************************************************************** #
# ********************* Sharif University of Technology ********************* #
# ****************** Department of Electrical Engineering ******************* #
# **************************** Deep Learning Lab **************************** #
# ************************ Video Synopsis Version 1.0 *********************** #
# *************** Authors: Ramtin Malekpoor - Mehrdad Morsali *************** #
# ********** ramtin.malekpour3@gmail.com - mehrdadmorsali@gmail.com ********* #
# *************************************************************************** #


# *************************************************************************** #
# ************************** Packages and Libraries ************************* #
# *************************************************************************** #
import numpy as np
import glob
import os
import cv2

def standardize_box(box, margin):
    """Converts a box's coordinates into standard values while adding margins
    to the box"""
    # x1, y1 is the coordinate of the upper-left side of the box
    # x2, y2 is the coordinate of the bottom-right side of the box
    x1 = int(box[0] - margin)
    y1 = int(box[1] - margin)
    x2 = int(box[2] + margin)
    y2 = int(box[3] + margin)
    # All coordinates should be a positive value
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(0, x2)
    y2 = max(0, y2)
    # x-coordinates should be less than the frame's width
    x1 = min(frame_width, x1)
    x2 = min(frame_width, x2)
    # y-coordinates should be less than the frame's height
    y1 = min(frame_height, y1)
    y2 = min(frame_height, y2)
    return x1, y1, x2, y2

# *************************************************************************** #
# ******************************** Functions ******************************** #
# *************************************************************************** #
def extract_background(images):
    # Define a matrix to keep red-channel values of the images
    r_matrix = np.zeros((images[0].shape[0], images[0].shape[1], len(images)),
                        dtype=np.float16)
    # Define a matrix to keep green-channel values of the images
    g_matrix = r_matrix.copy()
    # Define a matrix to keep blue-channel values of the images
    b_matrix = r_matrix.copy()

    # Fill the defined matrices with appropriate values
    for idx, image in enumerate(images):
        r_matrix[:, :, idx] = image[:, :, 2]
        g_matrix[:, :, idx] = image[:, :, 1]
        b_matrix[:, :, idx] = image[:, :, 0]

    r_channel = np.nanmedian(r_matrix, axis=2)
    g_channel = np.nanmedian(g_matrix, axis=2)
    b_channel = np.nanmedian(b_matrix, axis=2)

    # Combine average red, green, blue values to form the background
    background = np.dstack((b_channel, g_channel, r_channel)).astype(np.uint8)
    # Return the extracted background
    return background

def remove_foreground(frame, box_list):
    """removes object boxes from an image"""
    background = frame.copy()
    background = np.array(background, dtype = np.float64)
    # remove object boxes from the background
    for box in box_list:
        # amend the possible coordinate errors in the box
        x1, y1, x2, y2 = standardize_box(box, 0)
        # replace the bounding box area with an empty area
        background[y1:y2, x1:x2, :] = np.nan
    return background


def extract_synopsis_background(samples_path):
    """extracts the background for the video synopsis"""
    # Define a variable to store background samples
    sample_images = []
    # Load samples stored in samples_path by the main.py
    for file in sorted(glob.glob(samples_path + '/*.npy')):
        sample_images.append(np.load(file))
    # Extract the background
    background = extract_background(sample_images)
    return background
