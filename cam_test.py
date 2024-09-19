import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np

# Define a video processor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # OpenCV video capture object
        self.cap = cv2.VideoCapture(0)

    def recv(self, frame):
        # Capture a frame from the OpenCV video capture object
        ret, img = self.cap.read()
        if not ret:
            return frame

        # Convert the frame to the format required by streamlit-webrtc
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = np.array(img)
        return frame

    def __del__(self):
        # Release the video capture object
        if self.cap is not None:
            self.cap.release()

# Streamlit app
st.title("Webcam Stream with OpenCV and streamlit-webrtc")

# Use webrtc_streamer to stream the video
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    async_mode=True
)

# Optionally display some information about the stream
if webrtc_ctx.video_processor:
    st.write("Webcam is active.")
else:
    st.write("Webcam is not active.")
