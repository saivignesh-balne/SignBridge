import streamlit as st
import cv2

def main():
    st.title("Test Webcam")

    # Button to start webcam
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam")
            return

        st.write("Webcam is running")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            st.image(frame, channels="BGR")

        cap.release()

if __name__ == "__main__":
    main()
