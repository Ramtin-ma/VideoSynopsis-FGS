# Video Synopsis: FGS Model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract
Video synopsis is an efficient method for condensing surveillance videos. This technique begins with the detection and tracking of objects, followed by the creation of object tubes. These tubes consist of sequences, each containing chronologically ordered bounding boxes of a unique object. To generate a condensed video, the first step involves rearranging the object tubes to maximize the number of non-overlapping objects in each frame. Then, these tubes are stitched to a background image extracted from the source video.

The lack of a standard dataset for the video synopsis task hinders the comparison of different video synopsis models. This paper addresses this issue by introducing a standard dataset, called SynoClip, designed specifically for the video synopsis task. SynoClip includes all the necessary features needed to evaluate various models directly and effectively.

Additionally, this work introduces a video synopsis model, called FGS, with low computational cost. The model includes an empty-frame object detector to identify frames empty of any objects, facilitating efficient utilization of the deep object detector. Moreover, a tube grouping algorithm is proposed to maintain relationships among tubes in the synthesized video. This is followed by a greedy tube rearrangement algorithm, which efficiently determines the start time of each tube. Finally, the proposed model is evaluated using the proposed dataset.

## Features
- **FGS Model**: Fast and efficient video synopsis generation.
- **Empty-Frame Object Detector**: Identifies frames without any objects to optimize the object detection process.
- **Tube Grouping Algorithm**: Maintains relationships among object tubes.
- **Greedy Tube Rearrangement**: Efficiently determines the start time for each object tube.

## SynoClip Dataset
We introduce the **SynoClip** dataset, a standard dataset specifically designed for the video synopsis task. It includes all the necessary features needed to evaluate various models directly and effectively.

### Download the Dataset
You can download the SynoClip dataset from the following Google Drive link:

[Download SynoClip Dataset](#your-google-drive-link-here)

## Installation

### Prerequisites
- Python 3.x
- [Other dependencies]

### Install Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
