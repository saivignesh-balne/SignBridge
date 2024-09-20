import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Computer Vision Streamlit Application")

# Define the video processor class
class VideoProcessor:
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        # Apply Canny edge detection and convert to BGR
        image = cv2.cvtColor(cv2.Canny(image, 100, 200), cv2.COLOR_GRAY2BGR)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Adding STUN server configuration
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Initialize WebRTC streamer with STUN server configuration
webrtc_streamer(
    key="demo", 
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION  # STUN server config added here
)
