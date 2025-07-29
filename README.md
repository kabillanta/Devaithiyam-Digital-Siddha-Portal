# Siddha Medicine Portal & Chatbot

This project is a web application dedicated to providing information about traditional Siddha medicine. It features a modern, user-friendly frontend and a powerful backend powered by a Retrieval-Augmented Generation (RAG) chatbot. The chatbot can answer user questions about diseases, symptoms, and remedies based on a knowledge base derived from Ministry of AYUSH-aligned information.

## ✨ Features

-   **Interactive Frontend**: A clean and responsive user interface built with HTML, CSS, and vanilla JavaScript.
-   **RAG Chatbot**: An intelligent chatbot that uses a knowledge base (`cleaned_chatbot.csv`) to provide accurate and context-aware answers.
-   **Conversational Memory**: The chatbot remembers the context of the conversation for natural follow-up questions.
-   **Dynamic Formatting**: The chatbot's responses are formatted using Markdown for better readability (e.g., bold text, bullet points).
-   **RESTful API**: A robust backend built with FastAPI, providing a simple `/chat` endpoint for the frontend to consume.

## 🛠️ Tech Stack

**Backend:**
-   **Framework**: FastAPI
-   **Language**: Python 3.9+
-   **LLM & RAG**: LangChain, Google Gemini (`gemini-1.5-flash-latest`)
-   **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
-   **Vector Store**: FAISS (Facebook AI Similarity Search)

**Frontend:**
-   **Markup/Styling**: HTML5, CSS3
-   **Interactivity**: Vanilla JavaScript
-   **Markdown Rendering**: `marked.js`
-   **Icons**: Font Awesome

## 📂 Project Structure


.
├── .venv/                  # Virtual environment directory
├── corpus/
│   └── cleaned_chatbot.csv # The cleaned knowledge base for the RAG model
├── .env                    # Environment variables (API keys)
├── .gitignore              # Files to be ignored by Git
├── main.py                 # The FastAPI backend application
├── chatbot.html            # The chatbot interface page
├── index.html              # The main landing page
├── script.js               # JavaScript for the chatbot frontend
├── style.css               # CSS styles for all pages
└── README.md               # This file


## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   A code editor like VS Code
-   API keys for:
    -   Google AI (for Gemini)
    -   Hugging Face Hub

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/siddha-chatbot.git](https://github.com/your-username/siddha-chatbot.git)
    cd siddha-chatbot
    ```

2.  **Create a Python virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. You can generate one by running `pip freeze > requirements.txt` after installing the packages listed below.)*
    - `fastapi`
    - `uvicorn[standard]`
    - `langchain`
    - `langchain-google-genai`
    - `langchain-community`
    - `langchain-huggingface`
    - `pandas`
    - `faiss-cpu`
    - `sentence-transformers`
    - `python-dotenv`

4.  **Create the environment file:**
    Create a file named `.env` in the root of the project and add your API keys:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
    ```

### Running the Application

1.  **Start the Backend Server:**
    Run the FastAPI application using Uvicorn from your terminal:
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be running at `http://127.0.0.1:8000`.

2.  **Launch the Frontend:**
    Simply open the `index.html` or `chatbot.html` file in your web browser. You can do this by right-clicking the file in your file explorer and selecting "Open with" your preferred browser.

## 🤖 How to Use

1.  Navigate to the **Chatbot** page.
2.  Type your question about symptoms or diseases into the input box.
3.  Press "Send" or hit Enter.
4.  The chatbot will respond based on its knowledge base, and you can ask follow-up questions.

## 📝 API Endpoint

The backend provides one main endpoint:

### `POST /chat`

-   **Summary**: Receives a question and chat history, returns the bot's answer.
-   **Request Body**:
    ```json
    {
      "question": "I have a cough and a fever",
      "chat_history": [
        ["user's previous question", "bot's previous answer"]
      ]
    }
    ```
-   **Response Body**:
    ```json
    {
      "answer": "The bot's generated answer in Markdown format.",
      "sources": [
        "The source text chunk used to generate the answer."
      ]
    }
    ```

## 📄 Data Source

The knowledge base for this chatbot is sourced from the `cleaned_chatbot.csv` file, which contains information aligned with traditional Siddha texts and guidelines from the Ministry of AYUSH, Government of India.

---

**Disclaimer**: This chatbot is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
