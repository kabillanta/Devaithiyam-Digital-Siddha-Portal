import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

try:
    df = pd.read_csv('./corpus/chatbot.csv')
    df['text'] = df.apply(lambda row: f"Disease: {row['Disease']}\nSymptom: {row['Sign and Symptoms']}\nRemedy: {row['Remedy']}", axis=1)
    loader = DataFrameLoader(df, page_content_column='text')
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from the CSV.")

except FileNotFoundError:
    print("Error: 'chatbot.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- 3. Split Documents into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
print(f"Split documents into {len(docs)} chunks.")

# --- 4. Create Embeddings and Vector Store ---
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Create the FAISS vector store from our documents and embeddings
vectorstore = FAISS.from_documents(docs, embeddings)
print("Vector store created successfully.")

# --- 5. Set up the LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite" , GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY "))
print("LLM loaded successfully.")

# --- 6. Create the RetrievalQA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), 
    return_source_documents=True
)
print("RAG chain created. The chatbot is ready!")

# --- 7. Chatbot Interaction Loop ---
while True:
    query = input("\nAsk a question about a disease or remedy (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    if query:
        result = qa_chain({"query": query})
        print("\nAnswer:", result["result"])