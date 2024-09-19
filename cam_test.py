import streamlit as st
import cv2

def main():
    st.title("Webcam Test")

    # Button to start webcam
    if st.button("Start Webcam"):
        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Failed to open webcam")
            return

        st.write("Webcam is running. Click 'Stop Webcam' to stop.")

        # Placeholder for video feed
        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()

            if not ret:
                st.error("Failed to grab frame")
                break

            # Display the video feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Stop button to break the loop and release the camera
            if st.button("Stop Webcam"):
                break

        cap.release()

if __name__ == "__main__":
    main()
