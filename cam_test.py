import streamlit as st
import tempfile
import base64

# Streamlit app
st.title("Video Recording in Streamlit")

# JavaScript code to handle video recording
video_recorder_js = """
<script>
let mediaRecorder;
let recordedChunks = [];

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    const video = document.querySelector("video");
    video.srcObject = stream;
    video.play();
    
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const downloadLink = document.createElement('a');
        downloadLink.href = url;
        downloadLink.download = 'recording.webm';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    };
    mediaRecorder.start();
}

function stopRecording() {
    mediaRecorder.stop();
}
</script>
"""

# Display video element and controls for recording
st.markdown("""
<video style="width: 100%;" autoplay></video>
<button onclick="startRecording()">Start Recording</button>
<button onclick="stopRecording()">Stop Recording</button>
""", unsafe_allow_html=True)

# Inject JavaScript code into the Streamlit app
st.components.v1.html(video_recorder_js, height=100)

# Optionally handle video file upload if needed
uploaded_file = st.file_uploader("Upload your recorded video", type=["webm"])
if uploaded_file:
    st.video(uploaded_file)
