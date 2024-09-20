import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import cv2
import math

# Define the HandDetector class or import it if it's a custom module
class HandDetector:
    def __init__(self, maxHands=1):
        # Initialize the hand detector here
        pass
    
    def findHands(self, img):
        # Implement hand detection logic here
        # Return a list of detected hands and the updated image
        return [], img

# HandDetector instance
detector = HandDetector(maxHands=1)

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_frame = None
        self.detector = detector

    def recv(self, frame):
        self.current_frame = frame.to_ndarray(format="bgr24")
        return frame
    
    def process_frame(self):
        if self.current_frame is None:
            return
        
        img = self.current_frame
        hands, img = self.detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            bg = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y-20:y + h+20, x-20:x + w+20]
            
            imgCropShape = imgCrop.shape
            
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((300 - wCal) / 2)
                bg[:, wGap:wCal + wGap] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((300 - hCal) / 2)
                bg[hGap:hCal + hGap, :] = imgResize
            
            # Display the cropped and resized image for debugging
            st.image(imgCrop, channels="BGR", caption="Cropped Hand Image")
            st.image(bg, channels="BGR", caption="Background with Resized Hand")

        # Display the full frame with detections
        st.image(img, channels="BGR", caption="Full Camera Feed")

def app():
    st.title("Sign Language Recognition")

    # Initialize the webrtc streamer
    webrtc_ctx = webrtc_streamer(key="sign-language-recognition", video_processor_factory=SignLanguageProcessor, media_stream_constraints={"video": True, "audio": False})

    # Access the video processor and process frames
    if webrtc_ctx.video_processor:
        while True:
            webrtc_ctx.video_processor.process_frame()
            st.time.sleep(0.1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    app()
