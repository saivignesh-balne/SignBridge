import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def process_frame(frame):
    # Example processing function: Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def main():
    st.title("Webcam Stream with Streamlit")

    # Capture video using st.camera_input
    video_file = st.camera_input("Capture Video")
    
    if video_file:
        # Read the video file as a stream
        video_bytes = video_file.read()
        
        # Create a file-like object from the bytes
        video_stream = io.BytesIO(video_bytes)
        
        # Convert the video stream to a video capture object
        video_capture = cv2.VideoCapture(video_stream)
        
        if not video_capture.isOpened():
            st.error("Failed to open video stream")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            # Process the frame
            processed_frame = process_frame(frame)
            
            # Display the processed frame
            st.image(processed_frame, channels="GRAY")

        video_capture.release()
    else:
        st.text("No video input yet. Please capture or upload a video.")

if __name__ == "__main__":
    main()
