import cv2
import streamlit as st

st.title('Webcam Test')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error('Failed to open webcam')
else:
    st.write('Webcam is open')

    # Capture frames from the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error('Failed to capture image')
            break

        st.image(frame, channels='BGR', use_column_width=True)

        if st.button('Stop'):
            break

    cap.release()
    cv2.destroyAllWindows()
