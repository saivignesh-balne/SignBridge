import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import base64
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the TensorFlow model
@st.cache_resource  # Cache the model to avoid reloading on every frame
def load_model():
    model = tf.keras.models.load_model('model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Class labels for predictions
CLASS_LABELS = ['hi', 'i love you', 'yes']  # Update with your actual class labels

# Supported Indian languages with their codes for gTTS and deep-translator
LANGUAGES = {
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

# Function to generate and play audio
def play_audio(text, lang_code):
    try:
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

        # JavaScript to play the audio
        audio_js = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        """
        st.components.v1.html(audio_js, height=0)
    except Exception as e:
        st.error(f"Error in play_audio: {e}")

# Function to calculate the bounding box for a hand
def get_bounding_box(hand_landmarks, img_shape):
    img_height, img_width, _ = img_shape
    x_min = img_width
    y_min = img_height
    x_max = 0
    y_max = 0

    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    width = x_max - x_min
    height = y_max - y_min

    return x_min, y_min, width, height

# Streamlit app
st.title("Sign Language Recognition with Translation and Speech")

# Initialize session state variables
if 'last_label' not in st.session_state:
    st.session_state.last_label = "No Hand Detected"
if 'play_audio' not in st.session_state:
    st.session_state.play_audio = False
if 'video_running' not in st.session_state:
    st.session_state.video_running = False

# Language selection
target_language = st.selectbox("Select a target language for speech:", list(LANGUAGES.keys()))
target_lang_code = LANGUAGES[target_language]

# Start and Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Video"):
        st.session_state.video_running = True
with col2:
    if st.button("Stop Video"):
        st.session_state.video_running = False

# Placeholder for displaying the video feed
video_placeholder = st.empty()

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load the TensorFlow model
model = load_model()

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

# Check if the camera is opened successfully
if not cap.isOpened():
    st.error("Error: Could not open camera.")
    st.stop()

# Main loop for video processing
while st.session_state.video_running:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture frame.")
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Get the bounding box for the hand
            x_min, y_min, width, height = get_bounding_box(hand_landmarks, frame.shape)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

        # Perform recognition after a delay
        if st.session_state.get("start_time") is None:
            st.session_state.start_time = time.time()

        elapsed_time = time.time() - st.session_state.start_time

        if elapsed_time >= 2:  # 2 seconds delay for recognition
            # Get the bounding box for the hands
            hand_landmarks = results.multi_hand_landmarks
            if len(hand_landmarks) == 1:
                hand = hand_landmarks[0]
                x, y, w, h = get_bounding_box(hand, frame.shape)
                img_crop = frame[y-20:y + h+20, x-20:x + w+20]
            elif len(hand_landmarks) == 2:
                hand1 = hand_landmarks[0]
                hand2 = hand_landmarks[1]
                x1, y1, w1, h1 = get_bounding_box(hand1, frame.shape)
                x2, y2, w2, h2 = get_bounding_box(hand2, frame.shape)
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                img_crop = frame[y_min-20:y_max+20, x_min-20:x_max+20]

            # Preprocess the image for the model
            img_resized = cv2.resize(img_crop, (256, 256))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)

            # Make a prediction
            predictions = model.predict(img_input)
            predicted_class = np.argmax(predictions, axis=1)
            label = CLASS_LABELS[predicted_class[0]]

            # Update the last label
            st.session_state.last_label = label

            # Play audio for the predicted label
            play_audio(label, target_lang_code)

            # Reset the timer
            st.session_state.start_time = None
    else:
        # No hands detected
        st.session_state.last_label = "No Hand Detected"

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {st.session_state.last_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in the Streamlit app
    video_placeholder.image(frame, channels="BGR", use_column_width=True)

# Release the camera when the video is stopped
if not st.session_state.video_running:
    cap.release()
    st.write("Video feed stopped.")
