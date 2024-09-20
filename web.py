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

# Streamlit app title
st.title("Sign Language Recognition with Translation and Speech")

# User input for language selection
language = st.selectbox("Select a target language for speech:", list(languages.keys()))
lang_code = languages[language]

# Define the audio playback function
def play_audio(text, lang_code):
    translated_text = GoogleTranslator(source='en', target=lang_code).translate(text)
    st.write(f"Translated Text: {translated_text}")

    tts = gTTS(translated_text, lang=lang_code)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    audio_bytes = audio_buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    """
    st.components.v1.html(audio_html, height=0)

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        # Initialize session state variables inside the class
        if 'model' not in st.session_state:
            try:
                st.session_state.model = tf.keras.models.load_model('model.h5')
                st.session_state.model.compile()  # Explicitly compile the model
                st.session_state.class_labels = ['hi', 'i love u', 'yes']
            except Exception as e:
                st.error(f"Error loading model: {e}")

        if 'detector' not in st.session_state:
            st.session_state.detector = HandDetector(maxHands=2)

        if 'last_label' not in st.session_state:
            st.session_state.last_label = "No Hand Detected"

        self.detector = st.session_state.detector
        self.model = st.session_state.model
        self.class_labels = st.session_state.class_labels
        self.last_label = st.session_state.last_label
        self.start_time = None
        self.recognition_delay = 1.5

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        hands, img = self.detector.findHands(image)

        if hands:
            if self.start_time is None:
                self.start_time = time.time()

            elapsed_time = time.time() - self.start_time

            if elapsed_time >= self.recognition_delay:
                if len(hands) == 1:
                    x, y, w, h = hands[0]['bbox']
                    imgCrop = image[y-20:y + h+20, x-20:x + w+20]
                    w_combined = w
                    h_combined = h

                elif len(hands) == 2:
                    x1, y1, w1, h1 = hands[0]['bbox']
                    x2, y2, w2, h2 = hands[1]['bbox']

                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1 + w1, x2 + w2)
                    y_max = max(y1 + h1, y2 + h2)
                    w_combined = x_max - x_min
                    h_combined = y_max - y_min

                    imgCrop = image[y_min-20:y_max+20, x_min-20:x_max+20]

                bg = np.ones((300, 300, 3), np.uint8) * 255

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

                img_preprocessed = cv2.resize(bg, (256, 256))
                img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
                img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

                yhat = self.model.predict(img_preprocessed)
                predicted_class = np.argmax(yhat, axis=1)
                label = self.class_labels[predicted_class[0]]

                if label != self.last_label:
                    self.last_label = label
                    # Speak the new label
                    play_audio(label, lang_code)
                
                self.start_time = None
            else:
                label = self.last_label
        else:
            label = self.last_label

        # Overlay text on the frame
        frame_with_text = image.copy()
        cv2.putText(frame_with_text, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame_with_text

# RTC Configuration for WebRTC
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Initialize WebRTC streamer with the updated argument
webrtc_streamer(
    key="sign-language-recognition",
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    mode=WebRtcMode.SENDRECV
)
