import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Load the TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Re-compile the model (optional, adjust if needed)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class_labels = ['hello', 'no', 'yes']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Define the size of the output image (256x256 pixels)
output_size = (256, 256)

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.label = "waiting"
        self.last_update_time = time.time()
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    def recv(self, frame):
        # Convert the frame to RGB (MediaPipe works with RGB)
        rgb_frame = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hand landmarks
        result = self.hands.process(rgb_frame)
        
        # If hands are detected, process the hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                img_h, img_w, _ = rgb_frame.shape
                x_coords = [lm.x * img_w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * img_h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Expand the bounding box slightly for better framing
                padding = 20
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, img_w)
                y_max = min(y_max + padding, img_h)

                # Crop the hand region from the frame
                hand_crop = rgb_frame[y_min:y_max, x_min:x_max]

                # Resize the cropped hand image to 256x256
                hand_resized = cv2.resize(hand_crop, output_size)

                # Prepare image for prediction
                img_preprocessed = cv2.resize(hand_resized, (256, 256))
                img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
                img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

                # Check if 2 seconds have passed since the last update
                current_time = time.time()
                if current_time - self.last_update_time >= 2.0:
                    yhat = model.predict(img_preprocessed)
                    predicted_class = np.argmax(yhat, axis=1)
                    self.label = class_labels[predicted_class[0]]
                    self.last_update_time = current_time

        # Convert the frame back to BGR for OpenCV drawing
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw the label on the frame
        cv2.putText(bgr_frame, f'Prediction: {self.label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Return the frame to display
        return bgr_frame

# Streamlit app
st.title('Hand Sign Recognition')

# Define the RTC configuration
rtc_configuration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Create the WebRTC app with the custom video processor and RTC configuration
webrtc_streamer(
    key="example",
    video_processor_factory=VideoTransformer,
    rtc_configuration=rtc_configuration,  # Pass the RTC configuration here
    media_stream_constraints={"video": {"facingMode": "user"}}
)
