import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Define a VideoTransformer class to process frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Example processing: Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return color_frame

# Streamlit app
def main():
    st.title("WebRTC Streamlit Example")

    # Use the updated webrtc_streamer API
    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer
    )

if __name__ == "__main__":
    main()
