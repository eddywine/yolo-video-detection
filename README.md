## YOLO Car Detection
To set up and run the script for car detection using YOLO attached below, follow the following steps:

### Step 1: Install the required dependencies
- Install the required dependencies in a virtual environment using:
    `pip install -r requirements.txt`

### Step 2: Run the script
To run the script, run the command:
    - `python yolo_tracking.py` 


## Expected Behavior
The expected behavior is that the script will download the YOLO model download the video and run the inference on the video in real time identifying the cars in the video. You should see the video running with annotations of the cars and objects in the video in a new frame if you run the Python script on the terminal.

## Output Files
The following are the output files that will be created when you run the above script:
    - `video.mp4` file for the downloaded video
    - `output_tracked_video1.mp4` file for the tracked video file
    - continuous real-time tracking of cars window with real-time display of bounding boxes and scores
    - 
