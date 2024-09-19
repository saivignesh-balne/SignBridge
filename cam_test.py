import streamlit as st
import cv2
import numpy as np

st.title("Webcam Test")

# Add a button to start webcam
if st.button('Start Webcam'):
    # Using Streamlit's built-in webcam component
    img = st.camera_input("Capture your photo")

    if img:
        # Convert the image to an OpenCV format
        img_array = np.array(img)
        # Display the image
        st.image(img_array, channels='BGR')
