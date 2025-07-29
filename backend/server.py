import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple
from contextlib import asynccontextmanager

# --- FastAPI Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. SETUP ---
load_dotenv()

# --- Lifespan function to load the model on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events for the FastAPI application.
    Loads the data, initializes the model, and creates the conversational chain.
    """
    print("🚀 Server starting up, loading chatbot model...")
    
    # --- Load Data and Create Vector Store ---
    try:
        df = pd.read_csv('../corpus/cleaned_chatbot.csv')
        df['page_content'] = df.apply(
            lambda row: f"Disease: {row['disease']}\nSymptoms: {row['sign_and_symptoms']}\nRemedy: {row['remedy']}",
            axis=1
        )
        documents = DataFrameLoader(df, page_content_column='page_content').load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # --- Initialize LLM ---
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        # --- CORRECTED PROMPT ---
        # The placeholder is now {input} instead of {question}
        prompt_template = """
        You are a knowledgeable and empathetic guide to Siddha medicine, providing information aligned with Ministry of AYUSH guidelines. Your goal is to provide a clear, helpful, and confident response based ONLY on the provided context.

        Follow these steps to structure your response:
        1. Acknowledge the user's symptoms in a friendly and caring tone.
        2. Based on the context from the provided official information, confidently state what condition the symptoms might relate to.
        3. Present the traditional remedies mentioned in the context in a clear and organized way. Use bullet points and bold formatting for remedy names. Briefly describe each remedy.
        4. Combine all of this into a single, easy-to-read, and non-repetitive answer.
        5. Conclude with a clear and caring disclaimer that this is for informational purposes and a professional medical consultation is essential for a formal diagnosis.

        Context:
        {context}

        Question:
        {input}

        Helpful and Structured Answer:
        """
        
        # The input_variables list now includes 'input'
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )

        # --- Create the chain for answering questions ---
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        Youtubeing_chain = create_stuff_documents_chain(llm, PROMPT)
        
        # Store the complete retrieval chain in the app's state
        app.state.qa_chain = create_retrieval_chain(retriever, Youtubeing_chain)

        print("✅ Chatbot model loaded successfully!")
        
    except FileNotFoundError:
        print("❌ Error: 'cleaned_chatbot.csv' not found. The API will not be able to answer questions.")
        app.state.qa_chain = None
        
    yield
    # --- Cleanup on shutdown ---
    print("🔌 Server shutting down.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple Medical Chatbot API",
    lifespan=lifespan
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 2. API ENDPOINT ---
class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []

@app.post("/chat", summary="Endpoint to chat with the bot")
def chat_with_bot(request: ChatRequest):
    """
    Receives a question and chat history, then returns the bot's answer.
    """
    qa_chain = app.state.qa_chain
    if not qa_chain:
        return {"answer": "Chatbot is not available. Please check server logs.", "sources": []}
    
    # The key passed to invoke must match what the chain expects ('input')
    result = qa_chain.invoke({
        "input": request.question,
        "chat_history": request.chat_history
    })
    
    # Format the sources
    sources = [doc.page_content for doc in result.get("context", [])]
    
    return {
        "answer": result["answer"],
        "sources": sources
    }
