from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import matplotlib.pyplot as plt

# Function to download video
def download_youtube_video(url, output_path='video.mp4'):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
    }
    subprocess.run(['yt-dlp', url, '-o', output_path])
    return output_path

def track_video(video_path):
    # Load the YOLO model
    model = YOLO("yolov8n.pt")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])
    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object
    output_path = "output_tracked_video1.mp4"  # Output video file path
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    
    plt.ion()  # Turn on interactive mode 
    
    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            scores = results[0].boxes.conf.cpu().tolist() 
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            annotated_frame = results[0].plot()
            # Plot the tracks and scores
            if track_ids:
                for box, track_id, score in zip(boxes, track_ids, scores):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  
                    if len(track) > 30:  # Retain 30 tracks for 30 frames
                        track.pop(0)
                    # Draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=2,
                    )
                    # Annotate with the detection score
                    label = f"ID: {track_id}, Score: {score:.2f}"
                    cv2.putText(annotated_frame, label, (int(x - w/2), int(y - h/2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Write the annotated frame to the output video
            out.write(annotated_frame)
            
            # Display the frame using `matplotlib`
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.draw()  
            plt.pause(0.001)  
            # Clear the plot for the next frame
            plt.clf()
            
        else:
            break
    
    # Release the video capture object 
    cap.release()
    out.release()
    plt.close('all') 
    return output_path

if __name__ == "__main__":
    # Download Video From Youtune
    video_url = "https://youtu.be/MNn9qKG2UFI?si=Pt6RE8dt17OV67ne"
    downloaded_video = download_youtube_video(video_url)
    
    # Track objects in the downloaded video
    tracked_video_path = track_video(downloaded_video)
    print(f"Tracked video saved at {tracked_video_path}")


