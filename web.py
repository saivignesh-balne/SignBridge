import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
import time

# Streamlit app
st.title("Sign Language Recognition")

# Initialize session state variables
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'model' not in st.session_state:
    try:
        st.session_state.model = tf.keras.models.load_model('Data/model.h5')
        st.session_state.class_labels = ['hi', 'i love u', 'yes']
    except Exception as e:
        st.error(f"Error loading model: {e}")
if 'detector' not in st.session_state:
    st.session_state.detector = HandDetector(maxHands=2)
if 'last_label' not in st.session_state:
    st.session_state.last_label = "No prediction yet"

# Buttons to control the webcam
start_button = st.button('Start Webcam')
stop_button = st.button('Stop Webcam')

if start_button and not st.session_state.webcam_running:
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if st.session_state.cap.isOpened():
        st.session_state.webcam_running = True
    else:
        st.error("Failed to open webcam")

if stop_button and st.session_state.webcam_running:
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.webcam_running = False

# Display for webcam feed and prediction
frame_placeholder = st.empty()
label_placeholder = st.empty()

if st.session_state.webcam_running:
    # Timer variables for sign recognition delay
    start_time = None
    recognition_delay = 1  # 1 seconds delay

    while st.session_state.webcam_running:
        ret, frame = st.session_state.cap.read()
        
        if not ret:
            st.error("Failed to grab frame")
            st.session_state.webcam_running = False
            break

        # Detect hands in the frame
        hands, img = st.session_state.detector.findHands(frame)

        if hands:
            if start_time is None:
                start_time = time.time()  # Start the timer
            
            elapsed_time = time.time() - start_time

            if elapsed_time >= recognition_delay:
                if len(hands) == 1:
                    # If only one hand is detected, use its bbox
                    x, y, w, h = hands[0]['bbox']
                    imgCrop = frame[y-20:y + h+20, x-20:x + w+20]
                    w_combined = w
                    h_combined = h

                elif len(hands) == 2:
                    # If two hands are detected, get bounding boxes of both hands
                    x1, y1, w1, h1 = hands[0]['bbox']
                    x2, y2, w2, h2 = hands[1]['bbox']

                    # Find the coordinates for the bounding box that includes both hands
                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1 + w1, x2 + w2)
                    y_max = max(y1 + h1, y2 + h2)
                    w_combined = x_max - x_min
                    h_combined = y_max - y_min

                    # Crop the area containing both hands
                    imgCrop = frame[y_min-20:y_max+20, x_min-20:x_max+20]

                # Create a blank background
                bg = np.ones((300, 300, 3), np.uint8) * 255

                # Calculate aspect ratio and resize the cropped image
                aspectRatio = h_combined / w_combined
                if aspectRatio > 1:
                    k = 300 / h_combined
                    wCal = math.ceil(k * w_combined)
                    imgResize = cv2.resize(imgCrop, (wCal, 300))
                    wGap = math.ceil((300 - wCal) / 2)
                    bg[:, wGap:wCal + wGap] = imgResize
                else:
                    k = 300 / w_combined
                    hCal = math.ceil(k * h_combined)
                    imgResize = cv2.resize(imgCrop, (300, hCal))
                    hGap = math.ceil((300 - hCal) / 2)
                    bg[hGap:hCal + hGap, :] = imgResize

                # Preprocess the image for the model
                img_preprocessed = cv2.resize(bg, (256, 256))
                img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
                img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

                # Make predictions
                yhat = st.session_state.model.predict(img_preprocessed)
                predicted_class = np.argmax(yhat, axis=1)
                label = st.session_state.class_labels[predicted_class[0]]

                # Update last label with the new prediction
                st.session_state.last_label = label

                # Reset the timer after prediction
                start_time = None
            else:
                # Use the last label instead of 'Waiting'
                label = st.session_state.last_label
        else:
            # No hands detected, show the last predicted label
            label = st.session_state.last_label

        # Update the Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        label_placeholder.text(f'Prediction: {label}')
else:
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

st.text("Press 'Start Webcam' to begin capturing video and 'Stop Webcam' to stop.")