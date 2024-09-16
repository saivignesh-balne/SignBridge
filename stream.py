import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import math

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define class labels according to your model's classes
class_labels = ['hi', 'i love u', 'yes']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

label = "Waiting..."
start_time = None
recognition_delay = 2  # 2 seconds delay

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        st.write("Failed to grab frame")
        break

    # Convert the frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time

        if elapsed_time >= recognition_delay:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, c = frame.shape

            # Create a blank background
            bg = np.ones((300, 300, 3), np.uint8) * 255

            # Preprocess the image for the model
            img_preprocessed = cv2.resize(bg, (256, 256))
            img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
            img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

            # Make predictions
            yhat = model.predict(img_preprocessed)
            predicted_class = np.argmax(yhat, axis=1)
            label = class_labels[predicted_class[0]]

            # Reset the timer after prediction
            start_time = None
    
    else:
        # Reset the timer if no hand is detected
        start_time = None
    
    # Display the result (label) on the frame, always visible
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show the main camera feed using Streamlit
    st.image(frame_rgb, channels="RGB")

    # Add a small delay to make the Streamlit app responsive
    time.sleep(0.1)

cap.release()
