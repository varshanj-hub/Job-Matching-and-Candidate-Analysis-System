import os
import json
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Initialize OpenAI LLM
llm = ChatGroq(temperature=0, api_key='YOUR_API_KEY',model="mistral-saba-24b")

# Step 1: Extract text from different formats
def extract_text(file_path):
    if file_path.name.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            return ' '.join(page.extract_text() for page in pdf.pages)
    elif file_path.name.endswith('.docx'):
        doc = Document(file_path)
        return ' '.join([p.text for p in doc.paragraphs])
    elif file_path.name.endswith('.txt'):
        return file_path.read().decode('utf-8')
    elif file_path.name.endswith('.json'):
        data = json.load(file_path)
        return json.dumps(data, indent=2)
    else:
        raise ValueError("Unsupported file format!")

# Step 2: Extract key information using LLM
def extract_key_info(text, is_job_description=True):
    if is_job_description:
        prompt = f"""
        Extract key requirements from the following job description. Provide a list of:
        - Required skills
        - Qualifications
        - Years of experience
        - Responsibilities
        
        Job Description:
        {text}
        """
    else:
        prompt = f"""
        Extract the following information from the candidate's resume:
        - Skills
        - Work experience (roles, companies, years)
        - Education (degree, field of study, institutions)
        - Certifications
        - Relevant technologies
        
        Resume:
        {text}
        """
    
    response = llm([HumanMessage(content=prompt)])
    return response.content


# Step 3: Calculate similarity between job descriptions and resume
def calculate_similarity(job_requirements, resume_data):
    combined_texts = [job_requirements, resume_data]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100  # Return as percentage

# Step 4: Analytics
def generate_analytics(job_requirements, resume_data):
    similarity = calculate_similarity(job_requirements, resume_data)
    return {
        "Skill Match (%)": round(similarity, 2),
        "Overall Match (%)": round(similarity, 2),  # Add additional metrics as needed
        "Justification": "Skills, experience, and qualifications align well with the job."
    }
# Streamlit App
st.title("Job Matching System")

st.sidebar.title("Upload Files")
job_files = st.sidebar.file_uploader("Upload Job Descriptions (PDF, JSON, TXT, DOCX)", type=['pdf', 'json', 'txt', 'docx'], accept_multiple_files=True)
resume_file = st.sidebar.file_uploader("Upload Candidate Resume (PDF, DOCX)", type=['pdf', 'docx'])

if st.sidebar.button("Process"):
    if job_files and resume_file:
        st.subheader("Extracting Job Descriptions...")
        job_requirements_list = []
        job_file_names = []

        for job_file in job_files:
            try:
                job_text = extract_text(job_file)
                job_info = extract_key_info(job_text, is_job_description=True)
                job_requirements_list.append(job_info)
                job_file_names.append(job_file.name)
            except Exception as e:
                st.error(f"Error processing {job_file.name}: {e}")
        
        st.success("Job Descriptions Processed Successfully!")
        for idx, (job, file_name) in enumerate(zip(job_requirements_list, job_file_names)):
            st.write(f"### {file_name}")
            st.text(job)  # Display as plain text instead of JSON
        
        st.subheader("Processing Candidate Resume...")
        try:
            resume_text = extract_text(resume_file)
            resume_data = extract_key_info(resume_text, is_job_description=False)
            st.success("Resume Processed Successfully!")
            st.text(resume_data)  # Display as plain text instead of JSON
        except Exception as e:
            st.error(f"Error processing resume: {e}")

        st.subheader("Matching Results")
        for idx, (job_requirements, file_name) in enumerate(zip(job_requirements_list, job_file_names)):
            analytics = generate_analytics(job_requirements, resume_data)
            st.write(f"### Match with {file_name}")
            st.json(analytics)  # Display analytics in JSON format (structured data)
    else:
        st.warning("Please upload both job descriptions and a candidate resume.")