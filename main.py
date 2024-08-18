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
import argparse
import os
import shutil
import time
import cv2
import tracemalloc
import numpy as np

#from sfsort.SFSORT import SFSORT
from SFSORT import SFSORT
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

from background import extract_background
from background import extract_synopsis_background
from tube import Tube, interpolate_missed_boxes, segment_objects
from synopsis import generate_synopsis_video

import psutil


# *************************************************************************** #
# ******************************** Functions ******************************** #
# *************************************************************************** #
def parse_arguments():
    """Takes input arguments from the interface running main.py"""
    # Create a parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--InPath', help='path to the input file',
                        default='', required=True, type=str)
    parser.add_argument('--IDPath', help='path to the app files',
                        default='', required=True, type=str)

    parser.add_argument('--Model', help='path to model.pt file',
                        default='yolov8n.pt', required=True, type=str)
    parser.add_argument('--ConfTH', help='detection score threshold',
                        default=0.1, type=float)
    parser.add_argument('--IoUTH', help='IOU threshold for NMS',
                        default=0.45, type=float)
    parser.add_argument('--Half', help='half precision inference',
                        default=False, type=bool)
    parser.add_argument('--Device', help='cuda device: 0/0,1,2,3/cpu',
                        default='cpu', type=str)
    parser.add_argument('--MaxDet', help='maximum detections count',
                        default=30, type=int)
    parser.add_argument('--Class', help='detection classes: 0/0 2 3',
                        default=0, type=int)
    parser.add_argument('--UseStride', help='object detection with stride 3',
                        default=False, type=bool)
    
    parser.add_argument('--Dynamic_tuning', help='Tracker',
                        default=True, type=bool)
    parser.add_argument('--Cth', help='Tracker',
                        default=0.7, type=float)
    parser.add_argument('--HighTH_m', help='Tracker',
                        default=0.1, type=float)
    parser.add_argument('--MatchTH1_m', help='Tracker',
                        default=0.05, type=float)
    parser.add_argument('--NewTH_m', help='Tracker',
                        default=0.1, type=float)
                        
    parser.add_argument('--HighTH', help='threshold for valid detection',
                        default=0.82, type=float)
    parser.add_argument('--LowTH', help='threshold for possible detection',
                        default=0.3, type=float)
    parser.add_argument('--MatchTH1', help='threshold for first association',
                        default=0.5, type=float)
    parser.add_argument('--MatchTH2', help='threshold for second association',
                        default=0.1, type=float)
    parser.add_argument('--NewTH', help='new track threshold',
                        default=0.7, type=float)
    parser.add_argument('--MarginTimeout', help='marginal lost track timeout',
                        default=20, type=int)
    parser.add_argument('--CenterTimeout', help='central lost track timeout',
                        default=30, type=int)
    parser.add_argument('--HorizontalMargin', help='horizontal margin',
                        default=10, type=int)
    parser.add_argument('--VerticalMargin', help='vertical margin',
                        default=30, type=int)
    parser.add_argument('--FrameWidth', help='width of video frames',
                        default=1440, type=int)
    parser.add_argument('--FrameHeight', help='height of video frames',
                        default=800, type=int)

    parser.add_argument('--Test', help='run the app in the test mode',
                        default=False, type=bool)
    # Return the received arguments
    return vars(parser.parse_args())


def make_temporary_directories():
    """Creates temporary directories for the app"""
    # Remove directories if they already exist
    if os.path.exists(samples_path):
        shutil.rmtree(samples_path)

    # Make new directories
    os.mkdir(samples_path)

    return 0


def TubeID(input_tube):
    return input_tube.id


def save_frame_as_background_sample(image, ID):
    """Saves an image in the specified path under the name ID"""
    filename = str(ID) + '.npy'
    np.save(os.path.join(samples_path, filename), image)
    return 0


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


def interpolate_box(box_2, box_1):
    """Bounding box interpolation for stride object detection"""
    z1 = (box_2[0] + 2 * box_1[0]) // 3
    z2 = (box_2[1] + 2 * box_1[1]) // 3
    z3 = (box_2[2] + 2 * box_1[2]) // 3
    z4 = (box_2[3] + 2 * box_1[3]) // 3
    q1 = (2 * box_2[0] + box_1[0]) // 3
    q2 = (2 * box_2[1] + box_1[1]) // 3
    q3 = (2 * box_2[2] + box_1[2]) // 3
    q4 = (2 * box_2[3] + box_1[3]) // 3
    return [z1, z2, z3, z4], [q1, q2, q3, q4]


def crop_image(image, box):
    image_crop = image[box[1]:box[3], box[0]:box[2], :]
    return np.array(image_crop, dtype=np.uint8)


# *************************************************************************** #
# ********************************* Classes ********************************* #
# *************************************************************************** #
class DotAccess(dict):
    """Provides dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BackgroundFIFO:
    """Provides a FIFO of background for the low-precision detector"""

    def __init__(self, size):
        # Determine the FIFO size
        self.size = size
        # Create a list with the user-defined size
        self.background_list = [None] * self.size
        # Reset the pointer to the list's new entry index
        self.replacement_candidate = 0

    def initialize(self, initial_image):
        """ Clears and initializes the FIFO"""
        # Initialize all FIFO blocks with the first background
        self.background_list = [initial_image] * self.size
        # Reset the pointer to the list's new entry index
        self.replacement_candidate = 0
        return True

    def add(self, image):
        """ Adds an image to the FIFO"""
        # Put the image in the address pointed by replacement_candidate
        self.background_list[self.replacement_candidate] = image
        # Update the pointer
        self.replacement_candidate += 1
        if self.replacement_candidate == self.size:
            self.replacement_candidate = 0
        return True

    def read(self):
        """ Returns all images added to the FIFO"""
        return self.background_list


# *************************************************************************** #
# **************************** Hyper-parameters ***************************** #
# *************************************************************************** #

''' ******************* Video Synopsis Hyper-parameters ******************* '''
# The number of background samples
BACKGROUND_SAMPLES_COUNT = 50
# Minimum tube length for valid tubes
MIN_TUBE_LENGTH = 100
# Bounding box cropping margin for foreground extraction
CROP_MARGIN = 40
# Distance threshold for conjunct tubes
DISTANCE_TH = 0.017
# Collision threshold for conjunct tubes
COLLISION_TH = 75
# Small area boxes overlooking threshold for tube grouping
AREA_TH = 600
# Small area boxes overlooking threshold for group arrangement
BOX_AREA_TH = 0
# Group count for group arrangement
GROUP_COUNT = 25
# Decay rate for group arrangement
DECAY_RATE = 0.97
# Conflict cost threshold  for group arrangement
CONFLICT_TH = 0.04

''' ************** The emptyframe detector Hyper-parameters ************** '''
# Use BACKGROUND_FIFO_SIZE of background samples to generate background
BACKGROUND_FIFO_SIZE = 10
# Background sampling occurs every BACKGROUND_REFRESH_TIME frames
BACKGROUND_REFRESH_TIME = 1000

# Static detector kernels
KERNEL_ERODE = np.ones((3, 3), np.uint8)
KERNEL_DIL = np.ones((9, 9), np.uint8)

# The minimum area of a box containing human
HUMAN_AREA_L = 30
# The maximum area of a box containing human
HUMAN_AREA_U = 1000
# The aspect ratio of a box containing human
HUMAN_ASPECT_RATIO = 1.2
# Determine the maximum period that a generated background is valid to use
VALID_BACKGROUND_PERIOD = 500
# Define a reduced size for the background image in the empty frame detector 
# to improve computational efficiency.
DOWN_RATE = 4
# Frame number of the last background that was generated
LAST_GENERATED_BACKGROUND = 0
# Min interval to save frames(ensures frames come from different time periods for background generation)
MIN_SAVE_INTERVAL = 300


''' ************* Tube objects' segmentation Hyper-parameters ************* '''
# The extra padding to avoid computational errors
EXTRA_PAD = 20
# The threshold value for segmentation masks
MASK_THRESHOLD = 40
# The area ratio for valid foreground mask identification
FOREGROUND_RATIO = 0.45
# The height ratio for valid foreground mask identification
HEIGHT_RATIO = 0.9

# *************************************************************************** #
# ******************************** Variables ******************************** #
# *************************************************************************** #

# The initial state of the object detector
emptyframe_detection = False

# The report file in the Test mode
report_file = False

''' ********************** Tube generation variables ********************** '''
# List of tubes
tube_list = []
# Skipped frames if stride is used
skipped_frame_1 = None
skipped_frame_2 = None

''' ****************** The emptyframe detector variables ***************** '''
# A FIFO to keep background samples
background_FIFO = BackgroundFIFO(BACKGROUND_FIFO_SIZE)
# A counter to indicate the time for background sampling
background_utilization_time = 0
# Frame number of the last background that was generated. 
last_generated_background  = 0
# Frame number of the last empty-frame saved for background generation
last_saved_emptyframe   = 0
# Flag to check the background FIFO is initialized.
background_fifo_initialized = False

''' ************************* Synopsis variables ************************** '''
# Synopsis background sampling period
sampling_period = 0
# Synopsis background sampling counter
background_counter = 0

# *************************************************************************** #
# *********************************** Main ********************************** #
# *************************************************************************** #

# Parse arguments
args = parse_arguments()

# Determine paths to folders used by the app
samples_path = os.path.join(args['IDPath'], 'samples')

# Make folders used by the app
make_temporary_directories()

# Load the input video
video_capture = cv2.VideoCapture(args['InPath'])

# Extract the video metadata
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define a reduced size for the background image in the empty frame detector 
# to improve computational efficiency.
DOWNSIZED_BACKGROUND_WIDTH = frame_width // DOWN_RATE
DOWNSIZED_BACKGROUND_HEIGHT = frame_height // DOWN_RATE

# Specify the address to the output video
output_path = os.path.join(args['IDPath'], 'Synopsis.mp4')
# Initialize the output video generator
video_synopsis = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 frame_rate, (frame_width, frame_height))

# Instantiate an object detector
model = YOLO(args['Model'], 'detect')

# Check for GPU availability
device = select_device(args['Device'])
# Devolve everything to selected devices
model.to(device)

# Package SFSORT arguments into the standard form
tracker_arguments = {"dynamic_tuning": args['Dynamic_tuning'], "cth": args['Cth'],
                      "high_th": args['HighTH'], "high_th_m": args['HighTH_m'], 
                      "match_th_first": args['MatchTH1'],
                      "match_th_first_m": args['MatchTH1_m'],
                      "match_th_second": args['MatchTH2'], "low_th": args['LowTH'],
                      "new_track_th": args['NewTH'], "new_track_th_m": args['NewTH_m'],
                      "marginal_timeout": args['MarginTimeout'],
                      "central_timeout": frame_rate,
                      "horizontal_margin": args['HorizontalMargin'],
                      "vertical_margin": args['VerticalMargin'],
                      "frame_width": args['FrameWidth'],
                      "frame_height": args['FrameHeight']}

# Package the tracker args
tracker_arguments = DotAccess(tracker_arguments)

# Instantiate a tracker
tracker = SFSORT(tracker_arguments) 

# Report preparation for the Test mode
if args['Test']:
    # Specify the address to the report file
    report_path = os.path.join(args['IDPath'], 'report.txt')
    # Open the report file in the write mode
    report_file = open(report_path, "w")
    # Start Monitoring the RAM usage
    tracemalloc.start()
    # Register the current time for execution time measurement
    current_moment = "Algorithm execution started at = " + str(time.time())
    report_file.write(current_moment)

# Determine the sampling period for video synopsis background generation
sampling_period = frames_count // BACKGROUND_SAMPLES_COUNT

# Initialize runtimes for each component  
yolo_runtime = 0  
emptyframe_runtime = 0  
tracking_runtime = 0

start = time.time()
# Run the algorithm
for frame_number in range(frames_count):
    # Read a frame
    success, frame = video_capture.read()
    
    # Print processing information every 100 frames
    if frame_number % 1000 == 0:
        print(f"Processing frame {frame_number} of {frames_count}")
        print(f"RAM Usage: {psutil.virtual_memory().percent}%")

    # Generate the video synopsis if everything is done
    if not success or frame_number == frames_count-1: #frames_count-1

        # Register the current time for execution time measurement
        tube_generation_finishing_moment = time.time()

        # Eliminate short tubes
        for tube in tube_list:
            if len(tube.images) < MIN_TUBE_LENGTH:
                tube_list.remove(tube)

        # Sort tubes
        tube_list.sort(key=TubeID)

        # Generate the background
        background = extract_synopsis_background(samples_path)

        # Fill the missed tube boxes with zero
        tube_list = interpolate_missed_boxes(tube_list)
        
        # Register the current time for execution time measurement
        segmention_start = time.time()
        
        # Extract the objects of each tube by segmentation
        tube_list = segment_objects(tube_list, background, EXTRA_PAD,
                                    MASK_THRESHOLD, FOREGROUND_RATIO,
                                    HEIGHT_RATIO)
                                    
        # Register the current time for execution time measurement
        segmention_end = time.time()
        
        # Register the current time for execution time measurement
        synopsis_starting_moment = time.time()

        # Generate the synopsis video
        generate_synopsis_video(tube_list, frame_rate, DISTANCE_TH,
                                COLLISION_TH, AREA_TH, GROUP_COUNT, DECAY_RATE,
                                CONFLICT_TH, BOX_AREA_TH, background,
                                video_synopsis)

        # Register the current time for execution time measurement
        synopsis_finishing_moment = time.time()

        # Report the results for the Test mode
        if args['Test']:
            # Register times for execution time measurement
            moment = "\nTube generation finishing moment = "
            moment += str(tube_generation_finishing_moment)
            report_file.write(moment)
            moment = "\nVideo Synopsis starting moment = "
            moment += str(synopsis_starting_moment)
            report_file.write(moment)
            moment = "\nVideo Synopsis finishing moment = "
            moment += str(synopsis_finishing_moment)
            report_file.write(moment)
            # Register the RAM usage
            ram_usage = "\nRAM Usage = " + str(tracemalloc.get_traced_memory())
            report_file.write(ram_usage)
        break

    # Save skipped frames in detection with strides
    if args['UseStride']:
        if frame_number % 3 == 1:
            skipped_frame_1 = np.array(frame)
            continue
        if frame_number % 3 == 2:
            skipped_frame_2 = np.array(frame)
            continue

    # Use the emptyframe object detector if the scene is empty of people
    if emptyframe_detection:
        
        # Register the current time for execution time measurement
        start_emptyframe = time.time()

        # Update the background if it has been a long time since the last update.
        if frame_number > last_generated_background  + VALID_BACKGROUND_PERIOD:
            # Read all Samples
            backgrounds = background_FIFO.read()
            # Generate a new background
            background_small = extract_background(backgrounds)
            # Update background frame time
            last_generated_background  = frame_number
           
        # Decrease the frame size to save computational resources
        frame_small = cv2.resize(frame, (DOWNSIZED_BACKGROUND_HEIGHT, DOWNSIZED_BACKGROUND_WIDTH),
                                 interpolation=cv2.INTER_NEAREST)
        # Find the foreground using absolute difference between images
        foreground = cv2.absdiff(background_small, frame_small)
        # Convert the foreground's color system to the grayscale
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        # Convert the foreground to a binary image
        _, foreground_binary = cv2.threshold(foreground, 15, 255,
                                             cv2.THRESH_BINARY)

        # Do morphological operations on the foreground
        foreground_erosion = cv2.erode(foreground_binary, KERNEL_ERODE,
                                       iterations=1)
        foreground_binary = cv2.dilate(foreground_erosion, KERNEL_DIL,
                                       iterations=1)
        # Find foreground contours
        contours, hierarchy = cv2.findContours(foreground_binary,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_TC89_L1)

        # Contours with specific area and aspect ratio contain human
        # Check for possible human presence in all contours
        for contour in contours:
            # Calculate the contour's area
            area = cv2.contourArea(contour)
            # Check area to detect human contour
            if HUMAN_AREA_L < area < HUMAN_AREA_U:
                # Convert the contour to an approximate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                # Calculate the object's aspect ratio
                aspect_ratio = w / h
                # Check the aspect ratio to make sure of human presence
                if aspect_ratio < HUMAN_ASPECT_RATIO:
                    # Start using high-precision detector
                    emptyframe_detection = False
                    break
        
        # Periodically, add the current human-free sample to the FIFO as
        # a new background sample. Also, save the frame as a sample for
        # background extraction in the video synopsis
        if background_utilization_time == BACKGROUND_REFRESH_TIME:
            # Add the current frame to the backgrounds FIFO
            frame_small = cv2.resize(frame, (DOWNSIZED_BACKGROUND_HEIGHT, DOWNSIZED_BACKGROUND_WIDTH),
                        interpolation=cv2.INTER_NEAREST)
            background_FIFO.add(frame_small.copy())
            # Reset the sampling counter
            background_utilization_time = 0
            # Save the frame for video synopsis background extraction
            save_frame_as_background_sample(frame, frame_number)
        else:
            # Increase the sampling counter's value
            background_utilization_time += 1
        
        # Record the current time for execution measurement  
        end_emptyframe = time.time()  

        # Accumulate emptyframe detector runtime
        emptyframe_runtime += end_emptyframe - start_emptyframe

    if not emptyframe_detection:
        # Object detector requires image dimensions to be multiples of
        # 32. So, resize the frame by zero-padding for compatibility
        # Right padding to reach the nearest multiple of 32 width
        padding_right = 0
        if frame_width % 32 != 0:
            padding_right = ((frame_width // 32) + 1) * 32 - frame_width
        # Bottom padding to reach the nearest multiple of 32 height
        padding_bottom = 0
        if frame_height % 32 != 0:
            padding_bottom = ((frame_height // 32) + 1) * 32 - frame_height
        # Zero-pad the frame
        frame_padded = cv2.copyMakeBorder(frame, 0, padding_bottom, 0,
                                          padding_right, cv2.BORDER_CONSTANT)
        
        # Record the start time for YOLO prediction 
        start_yolo = time.time()  

        # Run the YOLO model to find objects in the padded frame  
        prediction = model.predict(frame_padded,  
                                   imgsz=(frame_padded.shape[0], frame_padded.shape[1]),  
                                   conf=0.1, iou=0.45,  
                                   half=False, device=device,  
                                   max_det=30,  
                                   classes=0)  
                                   
        # Record the end time for YOLO prediction  
        end_yolo = time.time()  
        
        # Accumulate YOLO runtime
        yolo_runtime += end_yolo - start_yolo  

        # Exclude extra info from the predictions
        prediction_results = prediction[0].boxes.cpu().numpy()
        box_list = prediction_results.xyxy
        score_list = prediction_results.conf

        # Skip further analysis if detector didn't find anyone
        if len(prediction_results) == 0:
            # Reset the background counter value
            background_counter = 0
            # Save the frame for video synopsis background extraction
            if frame_number > last_saved_emptyframe  + MIN_SAVE_INTERVAL:
                save_frame_as_background_sample(frame, frame_number)
                last_saved_emptyframe  = frame_number
            
            # Prepare everything for the emptyframe detector
            frame_small = cv2.resize(frame, (DOWNSIZED_BACKGROUND_HEIGHT, DOWNSIZED_BACKGROUND_WIDTH),
                        interpolation=cv2.INTER_NEAREST)
            
            # Initialize the background FIFO if it is not initialized, or just save emptyframe.
            if background_fifo_initialized:
                background_FIFO.add(frame_small.copy())
            else:
                background_FIFO.initialize(frame_small.copy())
                background_fifo_initialized = True

                # Reset the background sampling counter
                background_utilization_time = 0

            # Use the emptyframe detector for further frames
            emptyframe_detection = True
            continue
                    
        # Periodically, save the frame's background as a background sample
        # Extract a background sample when reaching the sampling period
        if background_counter == sampling_period:
            # Extract a background sample by removing objects from the frame
            background = remove_foreground(frame, box_list)
            # Save the frame for video synopsis background extraction
            save_frame_as_background_sample(background, frame_number)
            # Reset the counter value
            background_counter = 0
        else:
            # Increase the counter value
            background_counter += 1
        
        # Record the start time for tracking  
        start_tracking = time.time()  

        # Pass the predictions to the tracker for updating tracks  
        tracks = tracker.update(box_list, score_list)  

        # Record the end time for tracking  
        end_tracking = time.time()  

        # Accumulate the total runtime for tracking  
        tracking_runtime += end_tracking - start_tracking
        
        # Skip further analysis if the tracker is not tracking anyone
        if len(tracks) == 0:
            continue
        
        # Add detections of the current frame to the tubes
        for track in tracks:
            # Convert the bounding box into the standard format
            bbox = standardize_box(track[0], 0)
            # Enlarge the box for motion estimation in segmentation
            bbox_pad = standardize_box(track[0], CROP_MARGIN)

            # Tube ID is same as the track ID
            tube_id = track[1]

            # Gather all tube IDs
            tube_id_list = [tube.id for tube in tube_list]

            # If the tube_list contains the track, update tubes
            if tube_id in tube_id_list:
                # Pop the tube to update it
                tube = tube_list.pop(tube_id_list.index(tube_id))

                # Append missed info in the case of using strides
                if args['UseStride']:
                    # Interpolate boxes from skipped frames
                    box1, box2 = interpolate_box(bbox, tube.boxes[-1])
                    # Add boxes to the tube
                    tube.add_box(box1)
                    tube.add_box(box2)
                    # Interpolate padded boxes
                    box1_pad, box2_pad = interpolate_box(bbox_pad,
                                                         tube.boxes_pad[-1])
                    # Add padded boxes to the tube
                    tube.add_box_pad(box1_pad)
                    tube.add_box_pad(box2_pad)
                    # Save frame number for interpolated boxes
                    tube.add_frame_number(frame_number - 1)
                    tube.add_frame_number(frame_number)
                    # Save the object's image
                    object_image = crop_image(skipped_frame_1, box1_pad)
                    tube.add_image(object_image)
                    object_image = crop_image(skipped_frame_2, box2_pad)
                    tube.add_image(object_image)
            else:
                # Create new tube
                tube = Tube(frame_number, tube_id)

            # Register the tube's bounding box
            tube.add_box(bbox)
            tube.add_box_pad(bbox_pad)
            # Register the last frame of the tube
            tube.set_last_frame(frame_number)
            # Save frame number
            tube.add_frame_number(frame_number + 1)
            # Save the object's image
            object_image = crop_image(frame, bbox_pad)
            tube.add_image(object_image)
            # Update Tube List
            tube_list.append(tube)

# Terminate the test mode
if args['Test']:
    # Close the report file
    report_file.close()
    # Stop Monitoring the RAM usage
    tracemalloc.stop()

# Close the input file
video_capture.release()

# Close the output file
video_synopsis.release()

# Print the runtime of each component
print("Yolo_Runtime:", yolo_runtime)
print("EmptyFrameDetector_Runtime:", emptyframe_runtime)
print("Tracking_Runtime:", tracking_runtime)
print("Segmention_Runtime:", segmention_end - segmention_start)
print("Synopsis_Runtime:", synopsis_finishing_moment - synopsis_starting_moment)
print("Total_Runtime:", synopsis_finishing_moment - start)
