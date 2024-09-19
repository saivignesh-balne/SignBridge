import cv2
import streamlit as st

st.title('Webcam Test')

for index in range(5):  # Try multiple indexes
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        st.write(f"Camera {index} is available.")
        ret, frame = cap.read()
        if ret:
            st.image(frame, channels='BGR', use_column_width=True)
        cap.release()
        break
    else:
        st.error(f"Camera {index} is not available.")
