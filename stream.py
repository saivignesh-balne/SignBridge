import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
import pyttsx3
import threading
import queue
import time

# Initialize Streamlit
st.title("Sign Language Recognition")

# Load the pre-trained model
model = tf.keras.models.load_model('Data/model.h5')

# Define class labels according to your model's classes
class_labels = ['hi', 'i love u', 'yes']

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Queue for text-to-speech
tts_queue = queue.Queue()

# Timer variables for sign recognition delay
start_time = None
recognition_delay = 2  # 2 seconds delay

# Function to speak the text
def speak(text):
    tts_queue.put(text)

def text_to_speech_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

# Start text-to-speech thread
tts_thread = threading.Thread(target=text_to_speech_worker)
tts_thread.start()

# Initialize the webcam and hand detector
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)

label = "Waiting..."

# Streamlit layout for the webcam feed
frame_display = st.empty()
prediction_display = st.empty()

# Quit flag
quit_flag = False

# Handle the quit button outside the loop
if st.button("Quit"):
    quit_flag = True

while cap.isOpened() and not quit_flag:
    ret, frame = cap.read()

    if not ret:
        st.write("Failed to grab frame")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(frame)

    if hands:
        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time

        if elapsed_time >= recognition_delay:
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

            # Speak the prediction
            speak(label)

            # Reset the timer after prediction
            start_time = None

    else:
        # Reset the timer if no hand is detected
        start_time = None

    # Display the result (label) on the Streamlit app
    prediction_display.text(f"Prediction: {label}")

    # Show the main camera feed using Streamlit
    frame_display.image(frame, channels="BGR")

# Release the capture and stop the TTS thread
cap.release()
tts_queue.put(None)
tts_thread.join()
