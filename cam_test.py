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
import tempfile
import threading

# Supported Indian languages with their codes for gTTS and deep-translator
languages = {
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
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
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
    st.session_state.last_label = "No Hand Detected"

# User input for text and language selection
language = st.selectbox("Select a target language for speech:", list(languages.keys()))
lang_code = languages[language]

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Display for video feed and prediction
frame_placeholder = st.empty()
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

def handle_predictions_and_audio(frame, detector, model, class_labels, lang_code):
    # Detect hands in the frame
    hands, img = detector.findHands(frame)

    if hands:
        start_time = time.time()
        recognition_delay = 2  # 2 seconds delay
        elapsed_time = time.time() - start_time

        if elapsed_time >= recognition_delay:
            if len(hands) == 1:
                x, y, w, h = hands[0]['bbox']
                imgCrop = frame[y-20:y + h+20, x-20:x + w+20]
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
                imgCrop = frame[y_min-20:y_max+20, x_min-20:x_max+20]

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
            yhat = model.predict(img_preprocessed)
            predicted_class = np.argmax(yhat, axis=1)
            label = class_labels[predicted_class[0]]

            # Speak the label
            play_audio(label, lang_code)
            
            return label
        else:
            return st.session_state.last_label
    else:
        return st.session_state.last_label

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.session_state.cap = cv2.VideoCapture(temp_file_path)

    while True:
        ret, frame = st.session_state.cap.read()

        if not ret:
            st.text("End of video or failed to grab frame.")
            break

        # Use threading to handle predictions and audio
        prediction_thread = threading.Thread(
            target=lambda: handle_predictions_and_audio(frame, st.session_state.detector, st.session_state.model, st.session_state.class_labels, lang_code)
        )
        prediction_thread.start()
        prediction_thread.join()

        # Update the Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        label_placeholder.text(f'Prediction: {st.session_state.last_label}')

    st.session_state.cap.release()
else:
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

st.text("Upload a video file to begin processing.")
