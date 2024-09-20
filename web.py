import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import cv2
from cvzone.HandTrackingModule import HandDetector

# Define the video processor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.7, maxHands=2)

    def transform(self, frame):
        # Convert the frame to RGB
        img = frame.to_ndarray(format="bgr24")

        # Flip the image for a later selfie-view display
        img = cv2.flip(img, 1)

        # Detect hands
        hands, img = self.detector.findHands(img, flipType=False)

        # Draw results
        if hands:
            for hand in hands:
                lmList = hand['lmList']
                bbox = hand['bbox']
                center = hand['center']
                handType = hand['type']

                # Draw the bounding box and landmarks
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                for lm in lmList:
                    cv2.circle(img, (lm[0], lm[1]), 5, (255, 0, 0), cv2.FILLED)

        return img

# Create a Streamlit app
st.title("Hand Detection with Streamlit and CVZone")

# Set up the webrtc stream
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
