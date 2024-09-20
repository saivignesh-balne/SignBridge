import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Computer Vision Streamlit application")

class VideoProcessor:
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(cv2.Canny(image, 100, 200), cv2.COLOR_GRAY2BGR)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="demo", video_processor_factory=VideoProcessor)
