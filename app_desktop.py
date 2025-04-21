import os
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
import time
from gtts import gTTS
from deep_translator import GoogleTranslator
import pygame
import threading
import tkinter as tk
from tkinter import ttk

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Supported Indian languages with their codes for gTTS and deep-translator
languages = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Bengali': 'bn',
    'Kannada': 'kn',
    'Gujarati': 'gu',
    'Marathi': 'mr',
    'Punjabi': 'pa',
    'Malayalam': 'ml'
}

# Load the model and initialize variables
try:
    model = tf.keras.models.load_model('model.h5')
    class_labels = ['hi', 'i love u', 'yes']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

detector = HandDetector(maxHands=2)
last_label = "No Hand Detected"

# Initialize delay variables
start_time = None
recognition_delay = 2  # 2 seconds delay between predictions

def play_audio(text, lang_code):
    # Translate the text
    translated_text = GoogleTranslator(source='en', target=lang_code).translate(text)
    print(f"Translated Text: {translated_text}")
    
    # Generate a unique filename for the audio file
    speech_file = f"speech_{time.time()}.mp3"
    
    # Convert the translated text to speech
    tts = gTTS(translated_text, lang=lang_code)
    tts.save(speech_file)
    
    # Play the audio file using threading
    def play_and_delete():
        pygame.mixer.music.load(speech_file)
        pygame.mixer.music.play()

        # Wait until the audio finishes playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # Unload the audio file and remove it after playing
        pygame.mixer.music.unload()
        if os.path.exists(speech_file):
            os.remove(speech_file)
            print(f"Deleted: {speech_file}")

    # Run the audio playback and deletion in a separate thread
    audio_thread = threading.Thread(target=play_and_delete)
    audio_thread.start()

def process_frame(frame, lang_code):
    global start_time, last_label
    
    # Detect hands in the frame
    hands, img = detector.findHands(frame)

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
            if (aspectRatio > 1):
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
            yhat = model.predict(img_preprocessed)
            predicted_class = np.argmax(yhat, axis=1)
            label = class_labels[predicted_class[0]]

            # Play the audio for the predicted label
            play_audio(label, lang_code)

            last_label = label
            start_time = None  # Reset the timer after the prediction
        else:
            label = last_label
    else:
        # Reset the timer if no hands are detected
        label = last_label

    return label, frame

def webcam_capture(lang_code):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        label, processed_frame = process_frame(frame, lang_code)

        # Show the processed frame and prediction in the console
        cv2.putText(processed_frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Webcam", processed_frame)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def on_language_select():
    # Get selected language from the dropdown
    selected_language = lang_dropdown.get()
    
    # Get the corresponding language code
    lang_code = languages[selected_language]
    
    # Start webcam capture with the selected language
    root.destroy()  # Close the tkinter window
    webcam_capture(lang_code)

# Create the tkinter window for language selection
root = tk.Tk()
root.title("Select Language")

# Create a dropdown menu (picker) for language selection
lang_label = ttk.Label(root, text="Select Language:")
lang_label.pack(pady=10)

lang_dropdown = ttk.Combobox(root, values=list(languages.keys()))
lang_dropdown.current(1)  # Default to Hindi
lang_dropdown.pack(pady=10)

# Create a button to start the webcam capture after selecting a language
start_button = ttk.Button(root, text="Start Webcam", command=on_language_select)
start_button.pack(pady=10)

# Start the tkinter mainloop to show the window
root.mainloop()
