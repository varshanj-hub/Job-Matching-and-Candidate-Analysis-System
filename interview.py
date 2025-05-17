import streamlit as st
from moviepy import VideoFileClip
import whisper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Initialize Whisper model
model = whisper.load_model("base")

# Function to process the uploaded video and extract audio
def process_video(uploaded_file):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the video file and extract audio
    video = VideoFileClip(temp_file_path)
    audio_file = "temp_audio.mp3"
    video.audio.write_audiofile(audio_file)

    return audio_file

# Function to transcribe audio and create analysis
def analyze_audio(audio_file):
    result = model.transcribe(audio_file)
    interview_text = result["text"]

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Split text into chunks for embedding
    chunks = interview_text.split('. ')
    vector_db = Chroma.from_texts(chunks, embedding_model)

    # Set up ChatGroq for analysis
    chat = ChatGroq(temperature=0, api_key='YOUR_API_KEY',model="mistral-saba-24b")

    # Prompt for analysis
    prompt = """
    You are an AI assistant tasked with analyzing an interview. Your response should be divided into the following sections:

    **Communication Style**: Assess the clarity and effectiveness of the candidate's communication. Mention if they are concise and how easy it is to understand their points.

    **Active Listening**: Evaluate the candidate's attentiveness and responsiveness to questions. Mention if they listened carefully, did not interrupt, and if their answers were relevant.

    **Engagement with the Interviewer**: Review how the candidate engaged with the interviewer. Include whether they responded promptly, demonstrated interest in the job, and built rapport.

    **Contextual Summary**: Provide an overall summary of the candidate's performance, highlighting key strengths and potential as a candidate for the job.

    Analyze the following interview segments and structure your response according to the sections above.
    """

    # Retrieve chunks for analysis
    retrieved_chunks = vector_db.similarity_search("Analyze the candidate's performance in terms of communication, listening, and engagement.")
    retrieved_texts = [doc.page_content for doc in retrieved_chunks]

    # Combine prompt and retrieved chunks
    contextual_prompt = f"{prompt}\n\nRetrieved Segments:\n" + "\n".join(retrieved_texts)

    # Get the AI's response
    response = chat.predict(contextual_prompt)
    return response

# Streamlit app
st.title("Interview Analysis App")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file (e.g., mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Process the video and extract audio
    audio_file_path = process_video(uploaded_file)

    # Display analysis
    st.write("Generating analysis of the Interview...")
    analysis_result = analyze_audio(audio_file_path)

    st.subheader("Analysis Result")
    st.write(analysis_result)
