import av
import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from streamlit_webrtc import webrtc_streamer

st.title("Computer Vision Streamlit Application")

# Define the video processor class
class VideoProcessor:
    def recv(self, frame):
        hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

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
