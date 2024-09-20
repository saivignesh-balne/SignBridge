import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import base64
import streamlit.components.v1 as components
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# Supported Indian languages with their codes for gTTS and deep-translator
languages = {
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

# RTC Configuration for WebRTC
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

class VideoTransformer(VideoTransformerBase):
    def __init__(self, lang_code):
        self.translated_text = "Hello"  # Default to "Hello" initially
        self.lang_code = lang_code

    def transform(self, frame):
        # Here you would handle video frame processing and detection
        # For now, just a placeholder for the functionality
        detected_label = "Hello"  # Placeholder for actual detection logic
        self.translated_text = GoogleTranslator(source='en', target=self.lang_code).translate(detected_label)
        return frame

def main():
    st.title("Translate and Automatically Play Speech in Indian Languages")

    # User input for language selection
    language = st.selectbox("Select a target language for speech:", list(languages.keys()))
    lang_code = languages[language]
    
    # Create a function that returns an instance of VideoTransformer with the selected language code
    def video_transformer_factory():
        return VideoTransformer(lang_code)

    # Display the webcam stream and process frames
    webrtc_ctx = webrtc_streamer(
        key="sign-language-recognition",
        video_processor_factory=video_transformer_factory,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        mode=WebRtcMode.SENDRECV
    )
    
    # Access the VideoTransformer instance via webrtc_ctx
    if webrtc_ctx.video_transformer:
        video_transformer = webrtc_ctx.video_transformer
        if video_transformer.translated_text:
            # Convert the translated text to speech in the selected language
            tts = gTTS(video_transformer.translated_text, lang=lang_code)

            # Use BytesIO to store the audio in memory
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)  # Move to the start of the BytesIO buffer

            # Get audio bytes and encode to base64 for embedding in HTML
            audio_bytes = audio_buffer.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                            <audio autoplay>
                                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            </audio>
                            <style>
                                audio {{
                                    display: none;
                                }}
                            </style>
                        """
            
            # Render the HTML audio player with autoplay
            st.components.v1.html(audio_html, height=50)
        else:
            st.warning("Waiting for label detection...")

if __name__ == "__main__":
    main()
