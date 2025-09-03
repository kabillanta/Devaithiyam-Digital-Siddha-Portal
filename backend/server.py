import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional
from contextlib import asynccontextmanager

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
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

# Helper function to classify diseases by name and symptoms
def classify_by_name_and_symptoms(disease_data):
    """
    Classify disease by name and symptoms when type is not available.
    """
    # Get the disease name and symptoms
    disease_name = str(disease_data.get('disease', '') or disease_data.get('name', '')).lower()
    symptoms = str(disease_data.get('sign_and_symptoms', '')).lower()
    remedy = str(disease_data.get('remedy', '')).lower()
    
    # Common keywords for each dosha type
    kabam_keywords = ['cough', 'cold', 'phlegm', 'mucus', 'respiratory', 'asthma', 'congestion', 'sinus', 'nausea']
    pitham_keywords = ['fever', 'heat', 'acid', 'inflammation', 'burning', 'gastritis', 'bile', 'rash', 'infection', 'vomiting']
    vatham_keywords = ['pain', 'arthritis', 'numbness', 'constipation', 'stiffness', 'dryness', 'cramp', 'gas', 'bloating', 'headache']
    
    # Check for keywords in name, symptoms, and remedy
    kabam_matches = sum(1 for word in kabam_keywords if word in disease_name or word in symptoms)
    pitham_matches = sum(1 for word in pitham_keywords if word in disease_name or word in symptoms)
    vatham_matches = sum(1 for word in vatham_keywords if word in disease_name or word in symptoms)
    
    # Assign type based on highest match count
    max_matches = max(kabam_matches, pitham_matches, vatham_matches)
    
    if max_matches == 0:
        disease_data['type'] = 'unknown'
    elif kabam_matches == max_matches:
        disease_data['type'] = 'kabam'
    elif pitham_matches == max_matches:
        disease_data['type'] = 'pitham'
    elif vatham_matches == max_matches:
        disease_data['type'] = 'vatham'
        
    return disease_data

# --- Lifespan function to load the model on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events for the FastAPI application.
    Loads the data, initializes the model, and creates the conversational chain.
    Also loads disease data categorized by dosha types.
    """
    print("🚀 Server starting up, loading chatbot model and disease data...")
    
    # --- Load Disease Data for Dosha Endpoints ---
    try:
        # Try multiple possible locations for the CSV file
        csv_paths = [
            'cleaned_chatbot_with_type.csv',  # Current directory
            '../corpus/cleaned_chatbot_with_type.csv',  # Corpus directory
            './cleaned_chatbot_with_type.csv'  # Explicit current directory
        ]
        
        dosha_df = None
        loaded_path = None
        
        for path in csv_paths:
            try:
                print(f"Trying to load CSV from: {path}")
                dosha_df = pd.read_csv(path)
                loaded_path = path
                break
            except Exception as e:
                print(f"Could not load from {path}: {str(e)}")
                continue
        
        if dosha_df is None:
            raise FileNotFoundError("Could not find the cleaned_chatbot_with_type.csv file in any expected location")
            
        print(f"✅ Loaded disease data from {loaded_path} with {len(dosha_df)} records")
        
        # Print column names to debug
        print(f"CSV columns: {dosha_df.columns.tolist()}")
        
        # Sample data for debugging
        print(f"Sample data (first 2 rows):")
        print(dosha_df.head(2))
        
        # Group diseases by dosha type
        kabam_diseases = []
        pitham_diseases = []
        vatham_diseases = []
        unknown_diseases = []
        
        # Check if 'type' column exists, if not, try to infer from disease names/symptoms
        has_type_column = 'type' in dosha_df.columns
        if not has_type_column:
            print("⚠️ No 'type' column found in CSV, will attempt classification based on disease names")
        
        # Process each row to categorize by dosha
        for _, row in dosha_df.iterrows():
            disease_data = row.to_dict()
            
            # Check if we have a type field
            if has_type_column and pd.notna(row['type']):
                disease_type = str(row['type']).lower().strip()
                
                # Categorize based on normalized type
                if disease_type in ['kabam', 'kapham', 'kapha']:
                    disease_data['type'] = 'kabam'  # Standardize type
                    kabam_diseases.append(disease_data)
                elif disease_type in ['pitham', 'pittam', 'pitta']:
                    disease_data['type'] = 'pitham'  # Standardize type
                    pitham_diseases.append(disease_data)
                elif disease_type in ['vatham', 'vatha', 'vata']:
                    disease_data['type'] = 'vatham'  # Standardize type
                    vatham_diseases.append(disease_data)
                else:
                    # If type doesn't match known doshas, try to classify by name/symptoms
                    classified = False
                    disease_data = classify_by_name_and_symptoms(disease_data)
                    
                    if disease_data['type'] == 'kabam':
                        kabam_diseases.append(disease_data)
                    elif disease_data['type'] == 'pitham':
                        pitham_diseases.append(disease_data)
                    elif disease_data['type'] == 'vatham':
                        vatham_diseases.append(disease_data)
                    else:
                        unknown_diseases.append(disease_data)
            else:
                # No type information, classify by name and symptoms
                disease_data = classify_by_name_and_symptoms(disease_data)
                
                if disease_data['type'] == 'kabam':
                    kabam_diseases.append(disease_data)
                elif disease_data['type'] == 'pitham':
                    pitham_diseases.append(disease_data)
                elif disease_data['type'] == 'vatham':
                    vatham_diseases.append(disease_data)
                else:
                    unknown_diseases.append(disease_data)
        
        # If any category is empty, redistribute some of the unknown diseases
        if len(kabam_diseases) == 0 or len(pitham_diseases) == 0 or len(vatham_diseases) == 0:
            print("⚠️ One or more dosha categories are empty, attempting to reclassify unknown diseases")
            
            # Try to classify unknown diseases more aggressively
            for disease in unknown_diseases[:]:  # Create a copy to iterate over
                disease_data = classify_by_name_and_symptoms(disease)
                if disease_data['type'] != 'unknown':
                    unknown_diseases.remove(disease)
                    if disease_data['type'] == 'kabam':
                        kabam_diseases.append(disease_data)
                    elif disease_data['type'] == 'pitham':
                        pitham_diseases.append(disease_data)
                    elif disease_data['type'] == 'vatham':
                        vatham_diseases.append(disease_data)
        
        # If we still have empty categories, distribute diseases evenly as a last resort
        if len(kabam_diseases) == 0 or len(pitham_diseases) == 0 or len(vatham_diseases) == 0:
            print("⚠️ Still have empty categories, distributing diseases evenly as fallback")
            
            # Collect all diseases
            all_diseases = kabam_diseases + pitham_diseases + vatham_diseases + unknown_diseases
            
            # Clear current categorization
            kabam_diseases = []
            pitham_diseases = []
            vatham_diseases = []
            unknown_diseases = []
            
            # Distribute evenly
            for i, disease in enumerate(all_diseases):
                category = i % 3
                if category == 0:
                    disease['type'] = 'kabam'
                    kabam_diseases.append(disease)
                elif category == 1:
                    disease['type'] = 'pitham'
                    pitham_diseases.append(disease)
                else:
                    disease['type'] = 'vatham'
                    vatham_diseases.append(disease)
        
        # Store the categorized diseases in app state
        app.state.kabam_diseases = kabam_diseases
        app.state.pitham_diseases = pitham_diseases
        app.state.vatham_diseases = vatham_diseases
        app.state.all_diseases = dosha_df.to_dict('records')
        
        print(f"Final dosha distribution:")
        print(f"- Kabam: {len(kabam_diseases)} diseases")
        print(f"- Pitham: {len(pitham_diseases)} diseases")
        print(f"- Vatham: {len(vatham_diseases)} diseases")
        print(f"- Uncategorized: {len(unknown_diseases)} diseases")
        
    except Exception as e:
        print(f"❌ Error loading dosha disease data: {str(e)}")
        # Initialize empty lists as fallback
        app.state.kabam_diseases = []
        app.state.pitham_diseases = []
        app.state.vatham_diseases = []
        app.state.all_diseases = []
    
    # --- Load Data and Create Vector Store for Chatbot ---
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
        Your response should be small and concise.
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
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        Youtubeing_chain = create_stuff_documents_chain(llm, PROMPT)
        
        # Store the complete retrieval chain in the app's state
        app.state.qa_chain = create_retrieval_chain(retriever, Youtubeing_chain)

        print("✅ Chatbot model loaded successfully!")
        
    except FileNotFoundError:
        print("❌ Error: 'cleaned_chatbot.csv' not found. The API will not be able to answer questions.")
        app.state.qa_chain = None
        
    yield
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


# --- 2. API MODELS ---
class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []

class Disease(BaseModel):
    disease: Optional[str] = None
    name: Optional[str] = None  # For compatibility with different CSV formats
    sign_and_symptoms: Optional[str] = None
    remedy: Optional[str] = None
    prepared_medicines: Optional[str] = None
    external_medicines: Optional[str] = None
    others: Optional[str] = None
    type: Optional[str] = None

class DiseaseResponse(BaseModel):
    count: int
    diseases: List[Dict]

# --- 3. API ENDPOINTS ---
@app.get("/", summary="Root endpoint")
def read_root():
    return {
        "message": "Siddha Medicine API",
        "endpoints": [
            "/diseases/kabam",
            "/diseases/pitham",
            "/diseases/vatham",
            "/diseases/all",
            "/diseases/search",
            "/chat"
        ]
    }

@app.get("/diseases/kabam", response_model=DiseaseResponse, summary="Get all Kabam diseases")
def get_kabam_diseases():
    """
    Returns all diseases classified under Kabam dosha.
    """
    diseases = app.state.kabam_diseases
    return {
        "count": len(diseases),
        "diseases": diseases
    }

@app.get("/diseases/pitham", response_model=DiseaseResponse, summary="Get all Pitham diseases")
def get_pitham_diseases():
    """
    Returns all diseases classified under Pitham dosha.
    """
    diseases = app.state.pitham_diseases
    return {
        "count": len(diseases),
        "diseases": diseases
    }

@app.get("/diseases/vatham", response_model=DiseaseResponse, summary="Get all Vatham diseases")
def get_vatham_diseases():
    """
    Returns all diseases classified under Vatham dosha.
    """
    diseases = app.state.vatham_diseases
    return {
        "count": len(diseases),
        "diseases": diseases
    }

@app.get("/diseases/all", response_model=DiseaseResponse, summary="Get all diseases")
def get_all_diseases():
    """
    Returns all diseases in the database.
    """
    diseases = app.state.all_diseases
    return {
        "count": len(diseases),
        "diseases": diseases
    }

@app.get("/diseases/search", response_model=DiseaseResponse, summary="Search for diseases")
def search_diseases(query: str):
    """
    Search for diseases by name or symptoms.
    """
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")
    
    query = query.lower()
    results = []
    
    for disease in app.state.all_diseases:
        # Check for disease name in different possible column names
        name = str(disease.get('disease', '') or disease.get('name', '')).lower()
        symptoms = str(disease.get('sign_and_symptoms', '')).lower()
        remedy = str(disease.get('remedy', '')).lower()
        
        if query in name or query in symptoms or query in remedy:
            results.append(disease)
    
    return {
        "count": len(results),
        "diseases": results
    }

@app.get("/remedies", summary="Get all remedies")
def get_remedies():
    """
    Returns all remedies from the remedy.csv file.
    """
    try:
        import os
        import csv
        
        # Possible paths for remedy.csv
        csv_paths = [
            'remedy.csv',  # Current directory
            '../backend/remedy.csv',  # Backend directory
            './remedy.csv'  # Explicit current directory
        ]
        
        remedies = []
        loaded_path = None
        
        for path in csv_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        remedies = list(reader)
                        loaded_path = path
                        break
            except Exception as e:
                continue
        
        if not remedies:
            raise FileNotFoundError("Could not find the remedy.csv file in any expected location")
        
        print(f"✅ Loaded remedies data from {loaded_path} with {len(remedies)} records")
        
        return {
            "count": len(remedies),
            "remedies": remedies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load remedies: {str(e)}")

@app.get("/remedies/search", summary="Search for remedies")
def search_remedies(query: str):
    """
    Search for remedies by name, preparation, or usage.
    """
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")
    
    try:
        # Get all remedies
        all_remedies_result = get_remedies()
        all_remedies = all_remedies_result["remedies"]
        
        query = query.lower()
        results = []
        
        for remedy in all_remedies:
            name = str(remedy.get("Remedy Name", "")).lower()
            preparation = str(remedy.get("Preparation", "")).lower()
            usage = str(remedy.get("Usage", "")).lower()
            
            if query in name or query in preparation or query in usage:
                results.append(remedy)
        
        return {
            "count": len(results),
            "remedies": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search remedies: {str(e)}")

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


