import streamlit as st

st.title("Webcam Example")

# Use Streamlit's camera input component
camera_input = st.camera_input("Capture an image")

if camera_input:
    st.image(camera_input)
else:
    st.write("No image captured yet.")
