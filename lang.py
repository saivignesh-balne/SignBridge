import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import base64

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

st.title("Translate and Automatically Play Speech in Indian Languages")

# User input for text and language selection
text_input = st.text_input("Enter an English word or phrase:")
language = st.selectbox("Select a target language for speech:", list(languages.keys()))

# Get the selected language code
lang_code = languages[language]

if text_input:
    # Translate the English input to the target language using deep-translator
    translated_text = GoogleTranslator(source='en', target=lang_code).translate(text_input)
    
    st.write(f"Translated Text: {translated_text}")
    
    # Convert the translated text to speech in the selected language
    tts = gTTS(translated_text, lang=lang_code)
    
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
    """
    
    # Render the HTML audio player with autoplay
    st.components.v1.html(audio_html, height=0)
else:
    st.warning("Please enter some text.")
