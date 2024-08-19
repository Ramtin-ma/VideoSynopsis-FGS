# *************************************************************************** #
# ********************* Sharif University of Technology ********************* #
# ****************** Department of Electrical Engineering ******************* #
# **************************** Deep Learning Lab **************************** #
# ************************ Video Synopsis Version 1.0 *********************** #
# *************** Authors: Ramtin Malekpour - Mehrdad Morsali *************** #
# ********** ramtin.malekpour3@gmail.com - mehrdadmorsali@gmail.com ********* #
# *************************************************************************** #


# *************************************************************************** #
# ************************** Packages and Libraries ************************* #
# *************************************************************************** #
import math
import numpy as np

import cv2
import numpy as np
# *************************************************************************** #
# ********************************* Classes ********************************* #
# *************************************************************************** #
class Tube:
    """Provides an object's temporal and spatial information"""

    def __init__(self, first_frame, tube_id):
        # Tube ID
        self.id = tube_id
        # The number of the first frame containing the object
        self.first_frame = first_frame
        # The number of the last frame containing the object
        self.last_frame = first_frame
        # The number of tube's first frame in the synopsis video
        self.first_frame_synopsis = 0
        # The number of tube's last frame in the synopsis video
        self.last_frame_synopsis = 0
        # The object's bounding boxes along the time
        self.boxes = []
        # The object's padded bounding boxes along the time
        self.boxes_pad = []
        # The number of all frames containing the object
        self.frame_number = []
        # A list of object images along the time
        self.images = []

    def set_last_frame(self, last_frame):
        """Registers the number of the last frame containing the object"""
        self.last_frame = last_frame
        self.last_frame_synopsis = len(self.boxes) - 1

    def add_box(self, box):
        """Adds a new box to the list of the object's bounding boxes"""
        self.boxes.append(box)

    def add_box_pad(self, box):
        """Adds a new box to the list of the object's padded bounding boxes"""
        self.boxes_pad.append(box)

    def add_image(self, image):
        """Adds a new image to the list of the object's images"""
        self.images.append(image)

    def add_frame_number(self, number):
        """Adds a new number to the list of frames containing the object"""
        self.frame_number.append(number)


# *************************************************************************** #
# ******************************** Functions ******************************** #
# *************************************************************************** #

def interpolate_missed_boxes(tube_list):
    """ Inserts zero in each tube's missed boxes spots"""
    for tube_index, tube in enumerate(tube_list):
        # Define lists that will contain missed information
        boxes_new = []
        boxes_pad_new = []
        images_new = []
        frames_new = []
        # Define a variable to indicate the old tube's positions
        original_index = 0
        # Check all tubes for missed information
        for new_index in range(tube.frame_number[0], tube.frame_number[-1]):
            # Add the missed info by zero-filling
            if new_index not in tube.frame_number:
                boxes_new += [[0, 0, 0, 0]]
                boxes_pad_new += [[0, 0, 0, 0]]
                images_new += [[0]]
            # Retain the present info
            else:
                boxes_new += [tube.boxes[original_index]]
                boxes_pad_new += [tube.boxes_pad[original_index]]
                images_new += [tube.images[original_index]]
                original_index += 1
            frames_new += [new_index]

        tube.boxes = boxes_new
        tube.boxes_pad = boxes_pad_new
        tube.images = images_new
        tube.frame_number = frames_new
        tube_list[tube_index] = tube

    return tube_list


def extract_object(image, box_height, height_ratio):
    """Extracts the desired object"""
    # Apply median filter to eliminate salt & pepper noise
    image_median = cv2.medianBlur(image, 3)
    # Pad the image to avoid computational errors
    image_pad = cv2.copyMakeBorder(image_median, 5, 5, 5, 5,
                                   cv2.BORDER_CONSTANT)
    # Apply the laplacian edge detector
    image_edge = cv2.Laplacian(image_pad, cv2.CV_8U)
    # Extract object contours
    contours, _ = cv2.findContours(image_edge, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Skip further operations if there is no object contour
    if len(contours) != 0:
        # Define the object's image
        image_mask = np.zeros(image_edge.shape, dtype=np.uint8)
        # The biggest contour delineates the object
        human = max(contours, key=cv2.contourArea)
        # Place the object on the black background
        cv2.drawContours(image_mask, [human], -1, (255, 255, 255), cv2.FILLED)
        # Remove the extra padding
        image_mask = image_mask[5:-5, 5:-5]
        # Reject the detected object if it is not human
        human = np.reshape(human, (-1, 2))
        # Find the bottom point of the human image
        bottom = np.amin(human, axis=0)[1]
        # Find the top point of the human image
        top = np.amax(human, axis=0)[1]
        # Find the height of the human image
        height = top - bottom
        ratio = height / box_height
        # The detection is valid when it shows height consistency with its box
        if ratio < height_ratio:
            image_mask = image
    else:
        image_mask = image
    return image_mask

def segment_objects(tube_list, background, pad=15, threshold_max=45,
                    foreground_ratio=0.48, height_ratio=0.9):

    """Modifies objects images in all tubes by segmentation"""
    for tube in tube_list:
        tube_length = len(tube.images)
        for idx in range(tube_length):
            # Skip segmentation if the image is empty
            image1 = tube.images[idx]
            
            # Load image's bounding boxes
            bbox1_pad = tube.boxes_pad[idx]
            bbox1 = tube.boxes[idx]
            
            # Continue for missing box
            if bbox1[2] < 1:
                continue
            
            image_slice = image1[bbox1[1]-bbox1_pad[1]:bbox1[3]-bbox1_pad[1]
                                ,bbox1[0]-bbox1_pad[0]:bbox1[2]-bbox1_pad[0]]

            # Do the segmentation for all images except the last one
            if idx < tube_length - 1:
                ''' **************** Find the motion mask ***************** '''
                bbox2_pad = tube.boxes_pad[idx + 1]
                
                # Only crop the image if the next image is empty
                image2 = tube.images[idx + 1]
                if bbox2_pad[2] < 1:
                    image_slice[image_slice < 1] = 1
                    tube.images[idx] = np.array(image_slice, dtype=np.uint8)
                    continue
                # The intersection of two boxes most likely contains the object
                

                # Find the top-left coordinate of the object's box
                x_tl = max(bbox1_pad[0], bbox2_pad[0])
                y_tl = max(bbox1_pad[1], bbox2_pad[1])
                # Find the bottom-right coordinate of the object's box
                x_br = min(bbox1_pad[2], bbox2_pad[2])
                y_br = min(bbox1_pad[3], bbox2_pad[3])

                # Find the dimensions of the object's box
                box_height = x_br - x_tl
                box_width = y_br - y_tl

                # Find the box's displacement
                delta_x = bbox2_pad[0] - bbox1_pad[0]
                delta_y = bbox2_pad[1] - bbox1_pad[1]

                # Find the first object's boundaries
                image1_x1 = max(0, delta_x)
                image1_y1 = max(0, delta_y)
                image1_x2 = image1_x1 + box_height
                image1_y2 = image1_y1 + box_width

                # Extract the first object
                object1 = image1[image1_y1:image1_y2, image1_x1:image1_x2, :]

                # Find the second object's boundaries
                image2_x1 = max(0, -delta_x)
                image2_y1 = max(0, -delta_y)
                image2_x2 = image2_x1 + box_height
                image2_y2 = image2_y1 + box_width

                # Extract the second object
                object2 = image2[image2_y1:image2_y2, image2_x1:image2_x2, :]

                # Calculate the motion
                motion = cv2.absdiff(object1, object2)
                motion = cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY)

                ''' ************** Find the foreground mask *************** '''
                # Extract the object's background
                background_slice = background[bbox1_pad[1]:bbox1_pad[3],
                                   bbox1_pad[0]:bbox1_pad[2], :]

                # Find the distance between the foreground and the background
                distance = cv2.absdiff(background_slice, image1)
                distance = cv2.cvtColor(distance, cv2.COLOR_BGR2GRAY)
                
                                 
                distance_slice = distance[bbox1[1]-bbox1_pad[1]:bbox1[3]-bbox1_pad[1]
                                ,bbox1[0]-bbox1_pad[0]:bbox1[2]-bbox1_pad[0]]

                ''' ***************** Find the final mask ***************** '''
                image_raw = np.array(distance_slice)
                # Combine the masks
                if (x_tl > bbox1[0]) or (x_br < bbox1[2]) or (y_tl > bbox1[1])\
                        or (y_br < bbox1[3]):
                    # The object's image before applying the thresholding mask
                    

                    # Find the dimensions of the object's box
                    box_width = min(bbox1[2], x_br) - max(bbox1[0], x_tl)
                    box_height = min(bbox1[3], y_br) - max(bbox1[1], y_tl)

                    # Find mask slice boundaries
                    start_x = max(0, x_tl - bbox1[0])
                    end_x = min(bbox1[2], x_br) - bbox1[0]
                    start_y = max(0, y_tl - bbox1[1])
                    end_y = min(bbox1[3], y_br) - bbox1[1]

                    # Modify the distance image
                    distance_slice = distance_slice[
                                     start_y:start_y + box_height,
                                     start_x:start_x + box_width]

                    # Modify the motion image
                    motion_x = max(0, bbox1[0] - x_tl)
                    motion_y = max(0, bbox1[1] - y_tl)
                    motion_slice = motion[motion_y:motion_y + box_height,
                                   motion_x:motion_x + box_width]

                    # The object's image assignment
                    selection = cv2.bitwise_or(distance_slice, motion_slice)                
                        
                    image_raw[start_y:end_y, start_x:end_x] = selection

                else:
                                   
                    motion_slice = motion[bbox1[1]-y_tl:bbox1[3]-y_tl
                                ,bbox1[0]-x_tl:bbox1[2]-x_tl]

                    # The object's image assignment
                    image_raw = cv2.bitwise_or(motion_slice, distance_slice)

                ''' **************** Apply the final mask ***************** '''
                # Pad image to avoid computational errors
                image_raw_pad = cv2.copyMakeBorder(image_raw, pad, pad, pad,
                                                   pad, cv2.BORDER_CONSTANT)

                # Denoise the image
                image_raw_pad = cv2.GaussianBlur(image_raw_pad, (5, 5), 0)

                # Find and apply the appropriate threshold value
                image_mask = np.zeros(image_raw_pad.shape, dtype=np.uint8)
                for threshold in range(threshold_max, 0, -5):
                    # Find an image mask
                    _, image_mask = cv2.threshold(image_raw_pad, threshold,
                                                  255, cv2.THRESH_BINARY)
                    # Find the ratio of foreground pixels
                    pixels_count = np.count_nonzero(image_mask == 255)
                    area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    ratio = pixels_count / area
                    # Stop when foreground pixels count is enough
                    if ratio >= foreground_ratio:
                        break

                # Modify the outlier mask pixels
                kernel = np.ones((5, 5), np.uint8)
                image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE,
                                              kernel, iterations=2)

                # Extract the exact object image
                image_mask = extract_object(image_mask, bbox1[3]-bbox1[1],
                                            height_ratio)

                # Modify the outlier mask pixels
                kernel = np.ones((15, 15), np.uint8)
                image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE,
                                              kernel, iterations=2)
                image_mask = cv2.blur(image_mask, (3, 3))

                # Remove the extra padding from the mask
                image_mask = image_mask[pad:- pad, pad: - pad]

                # Apply the mask
                image_slice[image_slice < 1] = 1
                image_slice[image_mask == 0] = 0
            else:
                image_slice[image_slice < 1] = 1

            tube.images[idx] = np.array(image_slice, dtype=np.uint8)

    return tube_list
