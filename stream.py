import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
import streamlit as st
from PIL import Image
import io

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')  # Change to 'my_model.keras' if needed

# Define class labels according to your model's classes
class_labels = ['Hi', 'I Love You', 'Yes']

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Function to capture frames from the webcam
def capture_frame():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()
    return frame

# Function to process and predict
def process_frame(frame):
    hands, img = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank background
        bg = np.ones((300, 300, 3), np.uint8) * 255
        imgCrop = frame[y-20:y + h+20, x-20:x + w+20]

        # Resize based on aspect ratio
        aspectRatio = h / w
        if aspectRatio > 1:
            k = 300 / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, 300))
            wGap = math.ceil((300 - wCal) / 2)
            bg[:, wGap:wCal + wGap] = imgResize
        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (300, hCal))
            hGap = math.ceil((300 - hCal) / 2)
            bg[hGap:hCal + hGap, :] = imgResize

        # Preprocess the image for the model
        img_preprocessed = cv2.resize(bg, (256, 256))
        img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        # Make predictions
        yhat = model.predict(img_preprocessed)
        predicted_class = np.argmax(yhat, axis=1)
        label = class_labels[predicted_class[0]]
    else:
        label = 'No hand detected'

    return label, img

# Streamlit app
st.title('Real-Time Hand Gesture Classification')

# Capture frame from webcam
frame = capture_frame()

# Process the frame
label, processed_frame = process_frame(frame)

# Convert the frame to a format suitable for Streamlit
def convert_to_streamlit_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# Display the result
st.image(convert_to_streamlit_image(processed_frame), caption='Hand Gesture', channels='BGR')
st.write(f'Prediction: {label}')

if st.button('Refresh'):
    # Refresh the frame
    frame = capture_frame()
    label, processed_frame = process_frame(frame)
    st.image(convert_to_streamlit_image(processed_frame), caption='Hand Gesture', channels='BGR')
    st.write(f'Prediction: {label}')
