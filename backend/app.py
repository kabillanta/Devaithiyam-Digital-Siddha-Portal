import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

try:
    df = pd.read_csv('./corpus/chatbot.csv')
    st.dataframe(df)
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
# print(f"Split documents into {len(docs)} chunks.")

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
from langchain.prompts import PromptTemplate

# --- Define the custom prompt for longer answers when context is found ---
prompt_template = """
You are a helpful medical assistant. Use the following pieces of context to answer the user's question in a detailed and explanatory manner.
Provide as much information as possible from the context, including symptoms and remedies if available.

Context:
{context}

Question:
{question}

Detailed Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Update your QA chain to use this prompt ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
print("RAG chain created. The chatbot is ready!")

# --- 7. Chatbot Interaction Loop ---
while True:
    query = input("\nAsk a question about a disease or remedy (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    if query:
        retrieved_docs = qa_chain.retriever.invoke(query)

        if retrieved_docs:
            print("✅ Found relevant info in the dataset. Generating a detailed answer...")
            result = qa_chain.invoke({"query": query})
            print("\nAnswer:", result["result"])

            # Optionally print the sources
            print("\nSources Used:")
            for doc in result["source_documents"]:
                print(f"- {doc.page_content[:100]}...") # Print first 100 chars

        else:
            print("⚠️ No specific info found in the dataset. Answering from general knowledge...")
            # 4. If no docs found, call the LLM directly
            general_response = llm.invoke(query)
            print("\nAnswer:", general_response.content)