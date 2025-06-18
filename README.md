# Job Matching and Candidate Analysis System

This project comprises two main components: a **Job Matching System** that intelligently matches resumes with job descriptions, and a **Candidate Analysis System** that provides an in-depth analysis of candidate interviews using video processing and AI.

## Features

* **Job Matching System**:
    * Extracts text from various document formats (PDF, DOCX, TXT, JSON) for job descriptions and resumes.
    * Utilizes an LLM (Language Model) to extract key requirements from job descriptions and relevant information from resumes (skills, experience, education, etc.).
    * Calculates similarity scores between job requirements and resume data to quantify the match.
    * Provides analytics on skill and overall match percentages with justifications.
* **Candidate Analysis System**:
    * Processes uploaded video interview files to extract audio.
    * Transcribes audio into text using the Whisper model.
    * Analyzes the transcribed interview text using an LLM to assess:
        * Communication Style
        * Active Listening
        * Engagement with the Interviewer
    * Generates a contextual summary of the candidate's performance.

## Technologies Used

* **Python**
* **Streamlit**: For creating interactive web applications. 
* **LangChain**: For building applications with large language models. 
* **HuggingFace Embeddings**: For creating text embeddings. 
* **ChromaDB**: For vector storage. 
* **Groq**: For fast inference with language models (ChatGroq). 
* **Whisper**: For audio transcription. 
* **MoviePy**: For video processing and audio extraction. 
* **pdfplumber, python-docx**: For document text extraction.
* **scikit-learn**: For TF-IDF vectorization and cosine similarity calculations.
* **python-dotenv**: For managing environment variables. 

## Setup and Installation

1.  **Clone the repository**:

    ```bash
    git clone <https://github.com/varshanj-hub/Job-Matching-and-Candidate-Analysis-System>
    cd <Job-Matching-and-Candidate-Analysis-System>
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r req.txt
    ```

4.  **Set up API Keys**:
    Create a `.env` file in the root directory of your project and add your Groq API key:

    ```
    GROQ_API_KEY='YOUR_GROQ_API_KEY'
    ```
    **Note**: You need to modify `interview.py` and `job.py` to correctly load the API key from the environment variable.

    * **For `interview.py`:**
        Replace `api_key='YOUR_API_KEY'` with `api_key=os.getenv('GROQ_API_KEY')`. 
        The line should look like this:
        ```python
        chat = ChatGroq(temperature=0, api_key=os.getenv('GROQ_API_KEY'), model="mistral-saba-24b")
        ```
    * **For `job.py`:**
        Replace `api_key='YOUR_API_KEY'` with `api_key=os.getenv('GROQ_API_KEY')`.
        The line should look like this:
        ```python
        llm = ChatGroq(temperature=0, api_key=os.getenv('GROQ_API_KEY'), model="mistral-saba-24b")
        ```

5.  **Install FFmpeg**:
    MoviePy requires FFmpeg. You can download it from the [FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's PATH, or install it via a package manager:

    * **On Ubuntu/Debian**:
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
    * **On macOS (using Homebrew)**:
        ```bash
        brew install ffmpeg
        ```
    * **On Windows**:
        Download the zip file from the FFmpeg website, extract it, and add the `bin` directory to your system's PATH environment variable.

## How to Run

This project consists of two separate Streamlit applications.

### Running the Job Matching System

To run the job matching system, navigate to the project's root directory and execute:

```bash
streamlit run job.py
