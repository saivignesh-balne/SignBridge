import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
import time
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import base64
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

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

# Streamlit app
st.title("Sign Language Recognition with Translation and Speech")

# Initialize session state variables
if 'model' not in st.session_state:
    try:
        st.session_state.model = tf.keras.models.load_model('model.h5')
        st.session_state.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        st.session_state.class_labels = ['hi', 'i love you', 'yes']
    except Exception as e:
        st.error(f"Error loading model: {e}")
if 'detector' not in st.session_state:
    st.session_state.detector = HandDetector(maxHands=2)
if 'last_label' not in st.session_state:
    st.session_state.last_label = "No Hand Detected"

# User input for text and language selection
language = st.selectbox("Select a target language for speech:", list(languages.keys()))
lang_code = languages[language]

# Display for webcam feed and prediction
label_placeholder = st.empty()

def play_audio(text, lang_code):
    # Translate the text
    translated_text = GoogleTranslator(source='en', target=lang_code).translate(text)
    st.write(f"Translated Text: {translated_text}")
    
    # Convert the translated text to speech
    tts = gTTS(translated_text, lang=lang_code)
    
    # Use BytesIO to store the audio in memory
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)  # Move to the start of the BytesIO buffer
    
    # Get audio bytes and encode to base64 for embedding in HTML
    audio_bytes = audio_buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    """
    
    # Render the HTML audio player with autoplay
    st.components.v1.html(audio_html, height=0)
    
def play_audio_thread(label, lang_code):
    # Run the audio generation and playback in a separate thread
    threading.Thread(target=play_audio, args=(label, lang_code)).start()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = None
        self.recognition_delay = 2  # 2 seconds delay

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detect hands in the frame
        hands, img = st.session_state.detector.findHands(img)

        if hands:
            if self.start_time is None:
                self.start_time = time.time()  # Start the timer
            
            elapsed_time = time.time() - self.start_time

            if elapsed_time >= self.recognition_delay:
                if len(hands) == 1:
                    # If only one hand is detected, use its bbox
                    x, y, w, h = hands[0]['bbox']
                    imgCrop = img[y-20:y + h+20, x-20:x + w+20]
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
                    imgCrop = img[y_min-20:y_max+20, x_min-20:x_max+20]

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

                # Speak the label
                play_audio_thread(label, lang_code)
                
                st.session_state.last_label = label
                
                # Reset the timer after prediction
                self.start_time = None
            else:
                label = st.session_state.last_label
        else:
            # Reset the timer if no hand is detected
            label = st.session_state.last_label

        # Update the Streamlit display
        label_placeholder.text(f'Prediction: {label}')
        return img

webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor)

st.text("Press 'Start Webcam' to begin capturing video and 'Stop Webcam' to stop.")
