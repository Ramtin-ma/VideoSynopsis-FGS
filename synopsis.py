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
import os
import cv2
import numpy as np
from scipy.signal import savgol_filter

from tube import Tube
import time

# *************************************************************************** #
# ******************************** Functions ******************************** #
# *************************************************************************** #
def weight_by_concurrency(x):
    """ Generates tube similarity weight based on concurrency """
    s = 1 + np.exp(x / 0.2)
    weight = (1 / s) + 1
    return weight ** 4


def group_tubes(tube_list, fps, distance_th, collision_th, area_th):
    """ Makes a list of conjunct tubes groups """
    tube_conjunction = []
    # Compare tube pairs
    for i in range(len(tube_list)):
        # Load the first tube
        tube1 = tube_list[i]
        for j in range(i + 1, len(tube_list)):
            # Load the second tube
            tube2 = tube_list[j]

            # Find the frames containing objects from both tubes
            common_frames = list(
                set(tube1.frame_number).intersection(tube2.frame_number))

            # Tubes are conjunct if they are concurrent for more than 1 seconds
            if len(common_frames) > fps:
                # Extract the first tube's boxes in common frames
                start = tube1.frame_number.index(min(common_frames))
                end = tube1.frame_number.index(max(common_frames))
                box_cf_1 = tube1.boxes[start:end]
                # Extract the second tube's boxes in common frames
                start = tube2.frame_number.index(min(common_frames))
                end = tube2.frame_number.index(max(common_frames))
                box_cf_2 = tube2.boxes[start:end]
                # Find the euclidean distance of box centers
                center_x_1 = (box_cf_1[:, 0] + box_cf_1[:, 2]) / 2
                center_y_1 = (box_cf_1[:, 1] + box_cf_1[:, 3]) / 2
                center_x_2 = (box_cf_2[:, 0] + box_cf_2[:, 2]) / 2
                center_y_2 = (box_cf_2[:, 1] + box_cf_2[:, 3]) / 2
                distance = np.sqrt((center_x_1 - center_x_2) ** 2 + (
                        center_y_1 - center_y_2) ** 2)

                # Find the dimensions of the intersection area
                max_x1 = np.maximum(box_cf_1[:, 0], box_cf_2[:, 0])
                max_y1 = np.maximum(box_cf_1[:, 1], box_cf_2[:, 1])
                min_x2 = np.minimum(box_cf_1[:, 2], box_cf_2[:, 2])
                min_y2 = np.minimum(box_cf_1[:, 3], box_cf_2[:, 3])
                height = min_x2 - max_x1
                width = min_y2 - max_y1

                # Convert the dimensions of non-overlapping boxes to zero
                height[height < 0] = 0
                width[width < 0] = 0

                # Calculate the area of each box
                area1 = np.multiply(box_cf_1[:, 2] - box_cf_1[:, 0],
                                    box_cf_1[:, 3] - box_cf_1[:, 1])
                area2 = np.multiply(box_cf_2[:, 2] - box_cf_2[:, 0],
                                    box_cf_2[:, 3] - box_cf_2[:, 1])

                # Find the minimum box area
                minimum_area = np.minimum(area1, area2)

                # Overlook the intersection for small boxes
                height[minimum_area < area_th] = 0
                width[minimum_area < area_th] = 0

                # Change the distance to undefined value for interpolated boxes
                distance[minimum_area == 0] = np.nan

                # Increase minimum area to avoid division by zero error
                minimum_area[minimum_area < 1] = 1

                # Calculate the average distance of tubes
                distance = np.divide(distance, minimum_area)
                average_distance = np.nanmean(distance)

                # Calculate the collision of tubes
                intersection_area = np.multiply(height, width)
                collision = np.sum(np.divide(intersection_area, minimum_area))

                # Calculate the weight based on concurrency
                concurrency = len(common_frames) / (
                    min(len(tube1.frame_number), len(tube2.frame_number)))

                weight = weight_by_concurrency(concurrency)

                # Apply weights
                average_distance = average_distance * weight
                collision = collision / weight

                # Register the conjunction scores for the tube pair
                tube_conjunction += [(i, j, average_distance, collision)]
            else:
                tube_conjunction += [(i, j, np.inf, 0)]

    # Find conjunct tube pairs
    conjunct_tubes = []
    for item in tube_conjunction:
        if item[2] < distance_th or item[3] > collision_th:
            conjunct_tubes += [(item[0], item[1])]

    # Group conjunct tubes
    groups = []
    conjunct_tubes_count = len(conjunct_tubes)
    for idx in range(conjunct_tubes_count):
        # Read a conjunct tube pair
        tube_pair = conjunct_tubes[idx]

        # Don't add a tube from a group to another one
        skip_grouping = False
        for group in groups:
            if (tube_pair[0] in group) or (tube_pair[1] in group):
                skip_grouping = True
        if skip_grouping:
            continue

        # Put the conjunct tube pair in a group
        group = [tube_pair[0], tube_pair[1]]

        # Check for new group members in other conjunct tubes
        for iteration in range(conjunct_tubes_count):
            # Stop expanding groups if finding other conjunct tubes failed
            stop_expansion = True
            # Try to find conjunct tubes from other groups
            for j in range(idx + 1, conjunct_tubes_count):
                if conjunct_tubes[j][0] in group:
                    if conjunct_tubes[j][1] not in group:
                        group += [conjunct_tubes[j][1]]
                        stop_expansion = False
                if conjunct_tubes[j][1] in group:
                    if conjunct_tubes[j][0] not in group:
                        group += [conjunct_tubes[j][0]]
                        stop_expansion = False
            if stop_expansion:
                break

        groups += [group]

    # Define a variable to keep the IDs of tubes in a group
    group_list = []
    for group in groups:
        group_id = []
        # Register the IDs of the group tubes
        for tube in group:
            group_id += [tube_list[tube].id]

        group_list += [group_id]

    # Add non-grouped tubes to the group list
    for tube in tube_list:
        non_grouped = True
        # check if the tube is a member of a group
        for group in group_list:
            if tube.id in group:
                non_grouped = False
        if non_grouped:
            group_list += [[tube.id]]

    return group_list


def sort_and_adjust_groups(tube_list, group_list):
    """ Sorts the group list based on their start time in the source video
    and adjusts group tubes in the synopsis video based on the temporal precedence """
    # Define a list to find the first frames of a group
    group_first_frames = np.zeros((len(group_list), 2), dtype=np.int32)
    # Adjust group tubes by the temporal precedence
    for idx, group in enumerate(group_list):
        # The first frame in the original video for tubes of the group
        first_frames = []
        # Register the first frame of each tube of the group
        for tube_id in group:
            for tube in tube_list:
                if tube.id == tube_id:
                    first_frames += [tube.first_frame]
        # Find the group's temporal precedent tube
        group_first_frames[idx, 0] = idx
        group_first_frames[idx, 1] = min(first_frames)
        # Move the precedent tube to frame 0 and adjust other tubes
        for tube_id in group:
            for tube in tube_list:
                if tube.id == tube_id:
                    tube.first_frame_synopsis = tube.first_frame - \
                                                min(first_frames)
                    tube.last_frame_synopsis = tube.first_frame_synopsis + \
                                               len(tube.boxes) - 1

    # Sort the group list by the temporal precedence
    group_first_frames = group_first_frames[group_first_frames[:,1].argsort()]
    
    group_list_sort = []
    for group_index in group_first_frames:
        group_list_sort += [group_list[group_index[0]]]
    
    return tube_list, group_list_sort
    
    
def sort_groups_synopsis(group_list, tube_list):
  """
  Sorts a list of groups based on the start time in synopsis video.

  Args:
      group_list: A list of lists, where each sublist represents a group.
      tube_list: A list of objects (presumably representing tubes) with a `first_frame_synopsis` attribute.

  Returns:
      A new list of groups, sorted based on the first frame synopsis of their corresponding tubes.
  """

  # Initialize an array to store start frame and group index
  start_frame_and_group_index = np.zeros((len(group_list), 2), dtype=np.int32)

  # Iterate through each group in the list
  for group_index, group in enumerate(group_list):
    # Initialize start frame with inf 
    start_frame_group = np.inf

    # Iterate through each element(index) within the current group
    for element_index in group:
      # Get the corresponding tube object
      tube = tube_list[element_index]

      # Update start of the group if the tube's start time is sooner
      if tube.first_frame_synopsis < start_frame_group:
        start_frame_group = tube.first_frame_synopsis

    # Store the minimum synopsis and group index in the array
    start_frame_and_group_index[group_index, 0] = start_frame_group
    start_frame_and_group_index[group_index, 1] = group_index

  # Sort the array based on the their start time
  start_frame_and_group_index = start_frame_and_group_index[start_frame_and_group_index[:, 0].argsort()]

  # Create a new list to store the sorted groups
  sorted_group_list = []

  # Iterate through the sorted indices and append corresponding groups
  for group_index in start_frame_and_group_index:
    sorted_group_list.append(group_list[group_index[1]])

  return sorted_group_list

    
def arrange_tubes(tube_list, group_list, PGL, decay_rate, box_area_th, cost_th):
    """ Arranges tubes to optimize the synopsis video """
    # Find the maximum frame shift for tubes
    max_frame_shift = 0
    for tube in tube_list:
        if tube.last_frame_synopsis > max_frame_shift:
            max_frame_shift = tube.last_frame_synopsis

    # List each group's tube indices
    group_list_indices = []
    for group in group_list:
        group_idx = []
        for tube_id in group:
            for tube in tube_list:
                if tube.id == tube_id:
                    group_idx += [tube_list.index(tube)]
        group_list_indices += [group_idx]

    # Find the number of groups
    group_count_total = len(group_list_indices)
    group_count_processed = len(PGL)
    for i in range(group_count_processed,group_count_total):
        # Find the group's maximum tube length
        max_group_length_i = 0
        for idx in group_list_indices[i]:
            tube = tube_list[idx]
            if len(tube.boxes) > max_group_length_i:
                max_group_length_i = len(tube.boxes)
                
        # Sort PGL
        PGL = sort_groups_synopsis(PGL, tube_list)
        group_count_processed = len(PGL)
        for j in range(group_count_processed):
            # Find the group's maximum tube length
            max_group_length_j = 0
            for idx in PGL[j]:
                tube = tube_list[idx]
                if len(tube.boxes) > max_group_length_j:
                    max_group_length_j = len(tube.boxes)
            # Find the maximum tube length for two groups
            max_group_length = max(max_group_length_i, max_group_length_j)
            # Arrange tubes
            decay = 1
            while True:
                conflict_cost = 0
                # Calculate the conflict cost of group tubes
                for idx1 in group_list_indices[i]:
                    tube1 = tube_list[idx1]
                    # Find the tube's displacement
                    shift = tube1.first_frame_synopsis - tube1.first_frame
                    # Update the tube's frame numbers
                    new_frames1 = [x + shift for x in tube1.frame_number]
                    for idx2 in PGL[j]:
                        tube2 = tube_list[idx2]
                        # Find the tube's displacement
                        shift = tube2.first_frame_synopsis - \
                                tube2.first_frame
                        # Update the tube's frame numbers
                        new_frames2 = [x + shift for x in
                                       tube2.frame_number]

                        # Find common frames of two tubes
                        common_frames = list(
                            set(new_frames1).intersection(new_frames2))
                        
                        #print("common_frames:",len(common_frames))

                        # Calculate the conflict cost of two tubes
                        groups_cost = 0
                        if common_frames:
                            # List the 1st tube's boxes in common frames
                            start = new_frames1.index(min(common_frames))
                            end = new_frames1.index(max(common_frames))
                            box_cf_1 = tube1.boxes[start:end]

                            # List the 2nd tube's boxes in common frames
                            start = new_frames2.index(min(common_frames))
                            end = new_frames2.index(max(common_frames))
                            box_cf_2 = tube2.boxes[start:end]

                            # Find the intersection coordinates
                            max_x1 = np.maximum(box_cf_1[:, 0],
                                                box_cf_2[:, 0])
                            max_y1 = np.maximum(box_cf_1[:, 1],
                                                box_cf_2[:, 1])
                            min_x2 = np.minimum(box_cf_1[:, 2],
                                                box_cf_2[:, 2])
                            min_y2 = np.minimum(box_cf_1[:, 3],
                                                box_cf_2[:, 3])

                            # Calculate the intersection dimensions
                            height = min_x2 - max_x1
                            width = min_y2 - max_y1
                            # Invalidate negative dimensions
                            height[height < 0] = 0
                            width[width < 0] = 0

                            # Find the area of the smaller box
                            box1_area = np.multiply(
                                box_cf_1[:, 2] - box_cf_1[:, 0],
                                box_cf_1[:, 3] - box_cf_1[:, 1])
                            box2_area = np.multiply(
                                box_cf_2[:, 2] - box_cf_2[:, 0],
                                box_cf_2[:, 3] - box_cf_2[:, 1])

                            min_area = np.minimum(box1_area, box2_area)

                            # Overlook excessively small boxes
                            overlooking_mask = min_area < max(box_area_th,1)
                            height[overlooking_mask] = 0
                            width[overlooking_mask] = 0
                            # Avoid zero division
                            min_area[overlooking_mask] = 1
                        
                            # Calculate the intersection area
                            intersection = np.multiply(height, width)

                            # Calculate the tube pair's collision cost
                            tube_cost = np.divide(intersection, min_area)
                            tubes_cost = np.sum(tube_cost)
                            # Update the group pair's collision cost
                            groups_cost += tubes_cost / max_group_length

                        # Calculate the conflict cost
                        conflict_cost += groups_cost 
                    # Keep going if the cost is still low
                    if conflict_cost * decay > cost_th:
                        break
                        
                # Shift tubes to decrease the cost
                if conflict_cost * decay > cost_th:
                    step = 3
                    decay_weight = decay_rate
                    if conflict_cost > 5 * cost_th:
                        decay_weight = decay_rate ** 4
                        step = 15

                    # Shift tubes
                    for idx in group_list_indices[i]:
                        tube_list[idx].first_frame_synopsis += step
                        last_frame = tube_list[idx].first_frame_synopsis \
                                     + len(tube_list[idx].boxes) - 1
                        tube_list[idx].last_frame_synopsis = last_frame

                        if last_frame > max_frame_shift:
                            max_frame_shift = last_frame
                            decay = decay * decay_weight
                    continue
                
                break
                
        # Add processed group to the Processed_group_list
        PGL += [group_list_indices[i]]
           
    return tube_list,PGL

def generate_synopsis_video(tube_list, fps, distance_th, collision_th, area_th,
                            K, decay_rate, cost_th, box_area_th,
                            background, synopsis_video):
        
    # Form groups of conjunct tubes
    for tube in tube_list:
        tube.boxes = np.array(tube.boxes, dtype=np.float64)
    group_list = group_tubes(tube_list, fps, distance_th, collision_th,
                             area_th)

    # Sort the group list by time and adjust its tubes by time
    tube_list, group_list = sort_and_adjust_groups(tube_list, group_list)

    # Calculate the average tube length
    average_tube_length = 0
    for tube in tube_list:
        average_tube_length += len(tube.boxes)
    average_tube_length = average_tube_length / len(tube_list)

    # The frame shift in the synopsis video
    shift = 0
    # The tube list in the synopsis video
    synopsis_tube_list = []

    # processed group list
    PGL = []

    # Calculate the number of iterations based on the length of group_list  
    loop_count = max((len(group_list) // K), 1)  

    # Iterate over the number of loops  
    for i in range(loop_count):  
        # Determine the selection range for groups  
        if i == 0:  
            selection_start = int(i * K)  
            selection_end = int((i + 2) * K)  
        else:  
            selection_start = int((i + 1) * K)  
            selection_end = int((i + 2) * K)  

        # Select groups from the group list  
        selected_groups = group_list[selection_start:selection_end]  

        # Assign start and end frames to each tube in the selected groups  
        for group in selected_groups:  
            for tube_id in group:  
                for tube in tube_list:  
                    if tube.id == tube_id:  
                        tube.first_frame_synopsis += shift  
                        tube.last_frame_synopsis = tube.first_frame_synopsis + len(tube.boxes) - 1  
                        synopsis_tube_list += [tube]  

        # Arrange the tubes based on the selected groups  
        synopsis_tube_list, PGL = arrange_tubes(synopsis_tube_list,  
                                                 group_list[:selection_end],  
                                                 PGL, decay_rate,  
                                                 box_area_th, cost_th)  

        # Determine the last and first synopsis frames  
        last_synopsis_frame = 0  
        first_synopsis_frame = np.inf  
        for tube in synopsis_tube_list:  
            if tube.last_frame_synopsis > last_synopsis_frame:  
                last_synopsis_frame = tube.last_frame_synopsis  
            if tube.first_frame_synopsis < first_synopsis_frame:  
                first_synopsis_frame = tube.first_frame_synopsis  

        # Create a mask initialized to zeros for frames without detections  
        mask = np.zeros(last_synopsis_frame - first_synopsis_frame)  

        # Populate the mask with ones for frames with detections  
        for j in range(first_synopsis_frame, last_synopsis_frame):  
            for tube in synopsis_tube_list:  
                if tube.first_frame_synopsis <= j <= tube.last_frame_synopsis:  
                    index = int(j - first_synopsis_frame)  
                    mask[index] += 1  

        # Smooth the mask using the Savitzky–Golay filter  
        window_length = int(0.15 * len(mask))  
        start_range = np.array(window_length)  
        if window_length % 2 == 0:  
            window_length -= 1  
        smooth_mask = savgol_filter(mask, window_length, 1)  

        # Calculate thresholds based on the smoothed mask  
        max_box = int(np.max(smooth_mask) * 0.75)  
        mean_box = np.mean(smooth_mask)  
        threshold = (max_box + mean_box) / 2  

        # Determine the shift based on the smoothed mask and threshold  
        for j in range(start_range, len(smooth_mask)):  
            if smooth_mask[j] < threshold:  
                shift = max(0, int(j - 0.5 * average_tube_length))  
                break  

    # Define variables to indicate the range of synopsis video frames#
    last_synopsis_frame = 0
    first_synopsis_frame = np.inf
    # Define a variable to indicate the maximum tube id
    max_id = 0
    for tube in tube_list:
        # Find the range of synopsis video frames
        if tube.last_frame_synopsis > last_synopsis_frame:
            last_synopsis_frame = tube.last_frame_synopsis
        if tube.first_frame_synopsis < first_synopsis_frame:
            first_synopsis_frame = tube.first_frame_synopsis
        # Find the maximum tube id
        if tube.id > max_id:
            max_id = tube.id

    # Generate a random color for each tube
    colors = np.random.randint(0, 255, size=(max_id + 1, 3), dtype="uint8")

    # Convert boxes to list for further use
    for tube in synopsis_tube_list:
        tube.boxes = np.array(tube.boxes, dtype=np.int16)
        tube.boxes = [list(box) for box in tube.boxes]
        
    # Stitching and video generation
    for j in range(first_synopsis_frame, last_synopsis_frame+1):
        # Fill the empty frame with the background
        frame = np.array(background)
        # Fill the frame with tubes
        for idx in range(len(synopsis_tube_list)):
            tube = synopsis_tube_list[idx]
            if tube.first_frame_synopsis <= j <= tube.last_frame_synopsis:
                bbox = tube.boxes.pop(0)
                # Load the tube's image
                image = tube.images.pop(0)
                # Skip stitching interpolated boxes
                if bbox == [0, 0, 0, 0]:
                    continue
                
                # Stitch the image to the background
                frame_slice = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                stitched_frame = np.where(image < 1, frame_slice, image)
                frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = stitched_frame
                # Add the tube id to the frame
                color = [int(c) for c in colors[tube.id]]
                text = str(tube.id)
                center = ((3*bbox[0] + bbox[2]) // 4, int(bbox[3]))
                cv2.putText(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
        # Write the frame to the synopsis video
        synopsis_video.write(frame)

    return synopsis_tube_list, group_list
