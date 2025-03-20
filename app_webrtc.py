import streamlit as st
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

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
CLASS_LABELS = ['hello', 'no', 'yes']  # Update with your actual class labels

# Custom VideoProcessor to process frames
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.model = load_model()
        self.start_time = None
        self.recognition_delay = 2  # 2 seconds delay for recognition
        self.last_label = "No Hand Detected"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format (BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe

        # Process the frame with MediaPipe Hands
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            # Start the recognition timer
            if self.start_time is None:
                self.start_time = time.time()

            elapsed_time = time.time() - self.start_time

            # Perform recognition after the delay
            if elapsed_time >= self.recognition_delay:
                # Get the bounding box for the hands
                hand_landmarks = results.multi_hand_landmarks
                if len(hand_landmarks) == 1:
                    hand = hand_landmarks[0]
                    x, y, w, h = self.get_bounding_box(hand, img.shape)
                    img_crop = img[y-20:y + h+20, x-20:x + w+20]
                elif len(hand_landmarks) == 2:
                    hand1 = hand_landmarks[0]
                    hand2 = hand_landmarks[1]
                    x1, y1, w1, h1 = self.get_bounding_box(hand1, img.shape)
                    x2, y2, w2, h2 = self.get_bounding_box(hand2, img.shape)
                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1 + w1, x2 + w2)
                    y_max = max(y1 + h1, y2 + h2)
                    img_crop = img[y_min-20:y_max+20, x_min-20:x_max+20]

                # Preprocess the image for the model
                img_resized = cv2.resize(img_crop, (224, 224))  # Resize to (224, 224)
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_input = np.expand_dims(img_normalized, axis=0)

                # Make a prediction
                predictions = self.model.predict(img_input)
                predicted_class_idx = np.argmax(predictions, axis=1)
                label = CLASS_LABELS[predicted_class_idx[0]]

                # Debug: Print confidence scores for each class
                print("Confidence Scores:", predictions)
                print("Predicted Class:", label)

                # Update the last label
                self.last_label = label

                # Reset the timer
                self.start_time = None
        else:
            # No hands detected
            self.last_label = "No Hand Detected"

        # Display the prediction on the frame
        cv2.putText(img, f"Prediction: {self.last_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")  # Return the processed frame

    def get_bounding_box(self, hand_landmarks, img_shape):
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

        return x_min, y_min, x_max - x_min, y_max - y_min

# Streamlit app
st.title("Sign Language Recognition")

# Start the WebRTC streamer
webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    async_processing=True,
    rtc_configuration={  # Add ICE servers for WebRTC
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)