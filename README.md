# Job-Matching-and-Candidate-Analysis-System

Overview

The Job Matching and Candidate Analysis System is an AI-powered tool to streamline recruitment. It automates resume-job matching and video interview analysis using Python, Streamlit, and NLP tools. Ideal for HR professionals, it helps make data-driven hiring decisions efficiently.

Features





Job Matching:





Extracts skills, qualifications, and experience from job descriptions and resumes.



Supports PDF, DOCX, TXT, JSON formats.



Calculates match scores using TF-IDF and cosine similarity.



Interview Analysis:





Transcribes video interviews with OpenAI Whisper.



Analyzes communication, listening, and engagement using HuggingFace embeddings and ChatGroq.



Supports MP4, MOV, AVI, MPEG4 formats.



User Interface:





Streamlit-based web app for easy file uploads and result visualization.

Screenshots

Job Matching System




Upload job descriptions and a resume for matching.

Interview Analysis App




Upload a video to analyze interview performance.

Requirements





Python 3.8+



Dependencies (see requirements.txt):





openai-whisper



langchain-groq



langchain-huggingface



langchain-community



moviepy



ffmpeg



chromadb



streamlit



pdfplumber



python-docx



scikit-learn

Installation





Clone the Repository:

git clone https://github.com/your-username/job-matching-candidate-analysis.git
cd job-matching-candidate-analysis



Install Dependencies:

pip install -r requirements.txt



Set Up Environment Variables:





Create a .env file in the project root.



Add your ChatGroq API key:

GROQ_API_KEY=your-api-key-here

Usage

Job Matching System





Run the app:

streamlit run job.py



Open http://localhost:8501 in your browser.



Upload job descriptions (PDF, DOCX, TXT, JSON) and a resume (PDF, DOCX).



Click "Process" to view match results.

Interview Analysis App





Run the app:

streamlit run interview.py



Open http://localhost:8501 in your browser.



Upload a video (MP4, MOV, AVI, MPEG4).



View the interview analysis.
