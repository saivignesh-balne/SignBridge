import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Streamlit app
def main():
    st.title("Simple WebRTC Streamlit Example")

    # Create a simple WebRTC video streamer
    webrtc_streamer(
        key="simple_example",
        # No need for video_transformer_factory or other parameters
    )

if __name__ == "__main__":
    main()
