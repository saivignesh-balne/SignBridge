import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained model (replace 'your_model_path' with the actual model path)
model = tf.keras.models.load_model('your_model_path')

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_frame = None
        self.current_prediction = None

    def process_frame(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format (BGR)
        img_resized = cv2.resize(img, (64, 64))  # Resize to match model input size (if needed)

        # Normalize the image before feeding it into the model
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_resized)
        class_id = np.argmax(prediction)  # Get the class ID with the highest probability

        # Set the current prediction to the class ID
        self.current_prediction = class_id

        # Draw the prediction on the frame
        cv2.putText(img, f"Prediction: {self.current_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img  # Return the modified frame

def app():
    st.title("Real-Time Sign Language Recognition")

    # Set up webrtc video streamer
    webrtc_ctx = webrtc_streamer(key="sign-language-recognition", video_processor_factory=SignLanguageProcessor)

    # If a processor is active, display the prediction
    if webrtc_ctx.video_processor:
        st.write(f"Current Prediction: {webrtc_ctx.video_processor.current_prediction}")

if __name__ == "__main__":
    app()
