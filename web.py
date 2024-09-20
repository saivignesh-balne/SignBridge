import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from cvzone.HandTrackingModule import HandDetector

# Streamlit title
st.title("Hand Detection with cvzone")

# Initialize the HandDetector from cvzone
detector = HandDetector(maxHands=2, detectionCon=0.8)

# Define the video processor class
class VideoProcessor:
    def recv(self, frame):
        # Convert the frame to a numpy array (OpenCV format)
        image = frame.to_ndarray(format="bgr24")
        
        # Detect hands
        hands, image = detector.findHands(image)  # with drawing
        
        # If hands are detected, draw landmarks and bounding boxes
        if hands:
            for hand in hands:
                # Get information about each hand
                bbox = hand["bbox"]  # Bounding box info x, y, w, h
                lmList = hand["lmList"]  # List of landmarks
                center = hand["center"]  # Center of the hand
                handType = hand["type"]  # Left or Right hand
                # You can use this data to process further or just visualize
        
        # Return the processed frame
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Adding STUN server configuration for WebRTC
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Initialize the WebRTC streamer
webrtc_streamer(
    key="hand-detection", 
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION
)
