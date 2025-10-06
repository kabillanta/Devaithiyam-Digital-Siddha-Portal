import os
import pandas as pd
import math
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional, Any
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# SIMPLE MODE FLAG
# Set environment variable SIMPLE_MODE=1 to skip LangChain / embeddings / LLM
# and use only lightweight CSV lookups for herbs, remedies, and diseases.
# This helps when you "can't execute" the full project (no API keys, limited
# environment, or just want the old simpler behavior).
# ---------------------------------------------------------------------------
SIMPLE_MODE = os.getenv("SIMPLE_MODE", "0") == "1"

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv

# --- LangChain Imports ---
if not SIMPLE_MODE:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_community.document_loaders import DataFrameLoader  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
    from langchain.chains import create_retrieval_chain  # type: ignore

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
    print(f"🚀 Server starting up (SIMPLE_MODE={'1' if SIMPLE_MODE else '0'})...")
    print("Loading disease data & (optionally) embeddings/LLM...")
    
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
    
    # --- Load Data and (optionally) Create Vector Store for Chatbot ---
    try:
        df = pd.read_csv('../corpus/cleaned_chatbot.csv')
        if not SIMPLE_MODE:
            df['page_content'] = df.apply(
                lambda row: f"Disease: {row['disease']}\nSymptoms: {row['sign_and_symptoms']}\nRemedy: {row['remedy']}",
                axis=1
            )
            documents = DataFrameLoader(df, page_content_column='page_content').load()  # type: ignore
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)  # type: ignore
            docs = text_splitter.split_documents(documents)  # type: ignore
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # type: ignore
            vectorstore = FAISS.from_documents(docs, embeddings)  # type: ignore

            llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", google_api_key=os.getenv("GOOGLE_API_KEY"))  # type: ignore
            prompt_template = """
You are a knowledgeable, concise, and empathetic Siddha medicine assistant, following Ministry of AYUSH guidelines. Your answers should be clear, friendly, and strictly factual.

You receive:
(a) Prior chat history,
(b) Retrieved context from the knowledge base,
(c) The latest user question.

Instructions:
- If the user greets or chats casually like hi, hey, hello, etc (not a medical question), reply with a brief, friendly line—do NOT provide medical advice.
- If the user asks about a disease, symptoms, remedies, or Siddha concepts:
  1. Briefly acknowledge the question.
  2. Summarize any related conditions found in the provided context.
  3. List only the relevant traditional remedies from the context, using short bullet points (Herb / Form – one key use or benefit). Do NOT invent or guess remedies.
  4. If safety, dosage, or diagnosis is unclear, add: "Consult a qualified Siddha practitioner for personalized advice." (single-line disclaimer)

Rules:
- Never fabricate or guess herbs, remedies, or facts not present in the context.
- If the answer is not in the context, clearly state you do not have that information.
- Keep responses concise and easy to understand.

Prior Chat History:
{chat_history}

Retrieved Context:
{context}

User Question:
{input}

Answer:
"""
            PROMPT = PromptTemplate(  # type: ignore
                template=prompt_template, input_variables=["context", "input", "chat_history"]
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # type: ignore
            document_chain = create_stuff_documents_chain(llm, PROMPT)  # type: ignore
            app.state.qa_chain = create_retrieval_chain(retriever, document_chain)  # type: ignore
            print("✅ Advanced QA chain initialized.")
        else:
            app.state.qa_chain = None
            app.state.raw_disease_rows = df.to_dict('records')
            print("⚡ SIMPLE_MODE active: Skipped embeddings & LLM.")

        # Herb & Remedy preload (common to both modes)
        app.state.herbs_index = {}
        app.state.remedies_index = {}
        app.state.remedies_list = []
        try:
            herb_paths = ['herb.csv','../backend/herb.csv','./herb.csv']
            for path in herb_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        herbs_list = list(reader)
                        for h in herbs_list:
                            names = set()
                            en = h.get('Herb Name (English)', '')
                            if en: names.add(en.lower())
                            tamil = h.get('Tamil Name','')
                            if tamil: names.add(tamil.lower())
                            botanical = h.get('Botanical Name','')
                            if botanical: names.add(botanical.lower())
                            for n in names:
                                app.state.herbs_index.setdefault(n, []).append(h)
                    break
            remedy_paths = ['remedy.csv','../backend/remedy.csv','./remedy.csv']
            for path in remedy_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rem_list = list(reader)
                        app.state.remedies_list = rem_list
                        for r_ in rem_list:
                            name = r_.get('Remedy Name','')
                            if name:
                                app.state.remedies_index.setdefault(name.lower(), []).append(r_)
                    break
            print(f"✅ Preloaded {len(app.state.herbs_index)} herb keys & {len(app.state.remedies_index)} remedy keys.")
        except Exception as preload_e:
            print(f"⚠️ Herb/Remedy preload failed: {preload_e}")

        if not SIMPLE_MODE:
            print("✅ Chatbot model loaded successfully!")
        else:
            print("✅ SIMPLE_MODE ready (CSV lookup only).")

    except FileNotFoundError:
        print("❌ Error: 'cleaned_chatbot.csv' not found. Continuing with SIMPLE_MODE only.")
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

def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Infinity and non-JSON-safe scalars to JSON-safe values.
    - NaN/Inf -> None
    - Supports dict/list/tuple recursion
    """
    # Fast path for common container types
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_safe(v) for v in obj)

    # Scalars
    if obj is None:
        return None
    try:
        # Handle pandas/numpy NaN/NaT
        if pd.isna(obj):
            return None
    except Exception:
        # Some types are not supported by pd.isna; ignore
        pass

    if isinstance(obj, float):
        # NaN or +/- Inf
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


# --- 2. API MODELS ---
class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []  # List of (user, assistant) pairs
    mode: Optional[str] = None  # future extension

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
            "/herbs",
            "/herbs/search",
            "/remedies",
            "/remedies/search",
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

def _format_history(history: List[Tuple[str,str]]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for u,a in history[-8:]:  # last 8 turns
        lines.append(f"User: {u}\nAssistant: {a}")
    return "\n".join(lines)

def _detect_direct_lookup(question: str, herbs_index: Dict[str, Any], remedies_index: Dict[str, Any]) -> Dict[str, Any]:
    q = question.lower().strip()
    # Simple heuristics: if starts with or equals a herb/remedy name or contains phrases "tell me about" / "info on"
    trigger_phrases = ["tell me about", "info on", "information on", "what is", "explain", "details of"]
    cleaned = q
    for phrase in trigger_phrases:
        cleaned = cleaned.replace(phrase, '').strip()
    # Try exact match first
    if cleaned in herbs_index:
        return {"type": "herb", "items": herbs_index[cleaned], "matched": cleaned}
    if cleaned in remedies_index:
        return {"type": "remedy", "items": remedies_index[cleaned], "matched": cleaned}
    # Fuzzy contains: check each key if it's a distinct word in question
    for key in herbs_index.keys():
        if key in q and len(key) > 3:
            return {"type": "herb", "items": herbs_index[key], "matched": key}
    for key in remedies_index.keys():
        if key in q and len(key) > 3:
            return {"type": "remedy", "items": remedies_index[key], "matched": key}
    return {"type": None}

def _collect_referenced_remedies(texts: List[str], remedies_list: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Scan provided texts (question + answer draft) for remedy names and return unique matches with full data."""
    if not remedies_list:
        return []
    joined = " \n ".join(t.lower() for t in texts if t)
    seen = set()
    results = []
    for r_ in remedies_list:
        name = (r_.get('Remedy Name') or '').strip()
        if not name:
            continue
        name_low = name.lower()
        if len(name_low) < 4:
            continue
        if name_low in joined and name_low not in seen:
            seen.add(name_low)
            results.append({
                'name': name,
                'preparation': r_.get('Preparation',''),
                'usage': r_.get('Usage',''),
                'url': f"remedies.html?query={name.replace(' ', '%20')}"
            })
    return results

# -------------------- NEW ENHANCEMENT HELPERS --------------------
def _get_remedy_name(row: Dict[str, Any]) -> str:
    """Return a best-effort remedy name from common column variants."""
    for key in ("Remedy Name", "Remeedy Name", "Remedy"):
        val = row.get(key)
        if val:
            return str(val)
    return ""

def _match_diseases(question: str, diseases: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Return list of disease rows whose name appears as whole word in question."""
    q = question.lower()
    # If the whole question is just a greeting, skip disease matching entirely
    if _is_greeting(q.strip()):
        return []
    results = []
    for d in diseases:
        name = (d.get('disease') or d.get('name') or '').strip()
        if not name:
            continue
        name_l = name.lower()
        # Avoid matching extremely short names and reduce false positives (e.g. 'hi' vs 'hiccups')
        # Require length >=4 OR multi-word disease names. Perform whole-word containment check.
        if (len(name_l) >= 4 or ' ' in name_l) and (f" {name_l} " in f" {q} "):
            results.append(d)
            if len(results) >= 3:
                break
    return results

# --- Greeting Detection Helper ---
_GREETING_PATTERNS = [
    'hi', 'hello', 'hey', 'hai', 'vanakkam', 'good morning', 'good evening', 'good afternoon',
    'greetings', 'hey there', 'hello there'
]

def _is_greeting(question: str) -> bool:
    """Basic heuristic to detect if user input is a pure greeting / pleasantry.
    Returns True if the entire (stripped) input is a short greeting phrase, or starts with a greeting and has only punctuation after.
    """
    if not question:
        return False
    q = question.lower().strip()
    # Exact match
    if q in _GREETING_PATTERNS:
        return True
    # Remove trailing punctuation for match
    q_np = q.rstrip('!.?')
    if q_np in _GREETING_PATTERNS:
        return True
    # Starts with greeting and then only punctuation / whitespace
    for g in _GREETING_PATTERNS:
        if q.startswith(g):
            tail = q[len(g):].strip()
            if not tail or all(ch in '!?.' for ch in tail):
                return True
    return False

def _remedies_from_disease_rows(rows: List[Dict[str,Any]], remedies_index: Dict[str,List[Dict[str,Any]]], remedies_list: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Given disease rows, extract referenced remedies by name using remedy.csv list.
    We scan the 'remedy' field for any known remedy names.
    """
    if not rows or not remedies_list:
        return []
    # Build quick set of remedy names
    name_map = {}
    for r in remedies_list:
        nm = _get_remedy_name(r).strip()
        if nm:
            name_map.setdefault(nm.lower(), r)
    found = []
    seen = set()
    for row in rows:
        rem_field = (row.get('remedy') or row.get('Remedy') or '').lower()
        if not rem_field:
            continue
        for rem_name_l, rem_row in name_map.items():
            if rem_name_l and rem_name_l in rem_field and rem_name_l not in seen:
                seen.add(rem_name_l)
                found.append(rem_row)
                if len(found) >= 5:
                    return found
    return found

def _detect_ingredient_query(question: str) -> List[str]:
    """If user lists available ingredients (e.g. 'I only have milagu, chukku'), extract them.
    Returns list of ingredient tokens (lowercase) or empty list if not an ingredient style query."""
    q = question.lower()
    triggers = ["i only have", "i have only", "i just have", "i have only got", "only have"]
    if not any(t in q for t in triggers):
        return []
    # Extract part after trigger
    for t in triggers:
        if t in q:
            tail = q.split(t,1)[1]
            break
    else:
        tail = q
    # Split by commas / 'and'
    raw = [seg.strip() for part in tail.split(' and ') for seg in part.split(',')]
    # Filter short tokens & stopwords
    stop = {"i","have","only","just","got","with","me","now","today"}
    ingredients = [r for r in raw if len(r) > 2 and r not in stop]
    return ingredients[:6]

def _suggest_remedies_for_ingredients(ings: List[str], remedies_list: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Find remedies whose Preparation field contains ALL of the provided ingredient tokens.
    Simple substring containment; returns up to 5 matches."""
    if not ings or not remedies_list:
        return []
    matches = []
    for r in remedies_list:
        prep = (r.get('Preparation') or '').lower()
        if all(ing.lower() in prep for ing in ings):
            matches.append(r)
            if len(matches) >= 5:
                break
    return matches

# ---------- Filter Search helpers ----------
_ING_SYNONYMS: Dict[str, List[str]] = {
    # Tamil -> English/common
    "milagu": ["pepper"],
    "inji": ["ginger"],
    "chukku": ["dry ginger", "dried ginger"],
    "seeragam": ["cumin"],
    "jeeragam": ["cumin"],
    "poondu": ["garlic"],
    "manjal": ["turmeric"],
    "thulasi": ["tulsi", "holy basil"],
    "adathodai": ["adhatoda", "vasaka", "justicia adhatoda"],
}

def _normalize_ingredient_tokens(raw: str) -> List[str]:
    if not raw:
        return []
    # split on comma and whitespace, keep tokens >= 2 chars
    parts: List[str] = []
    for chunk in raw.split(','):
        parts.extend(chunk.strip().split())
    tokens = [p.strip().lower() for p in parts if len(p.strip()) >= 2]
    return tokens[:12]

def _expand_token_variants(tokens: List[str]) -> Dict[str, List[str]]:
    variants: Dict[str, List[str]] = {}
    for t in tokens:
        # include the token itself and known synonyms/phrases
        v = [t]
        v.extend(_ING_SYNONYMS.get(t, []))
        variants[t] = list(dict.fromkeys(v))  # dedupe preserve order
    return variants

def _prep_matches_variants(prep: str, token_variants: Dict[str, List[str]], mode: str) -> Dict[str, Any]:
    """Return match info for a preparation text against ingredient token variants.
    mode: 'all' -> all tokens must match at least one variant; 'any' -> any token matches.
    Returns { matched: [tokens], missing: [tokens], match_count: int }
    """
    prep_low = (prep or '').lower()
    matched: List[str] = []
    missing: List[str] = []
    for base, vars_ in token_variants.items():
        hit = any(v in prep_low for v in vars_)
        if hit:
            matched.append(base)
        else:
            missing.append(base)
    if mode == 'any':
        ok = len(matched) > 0
    else:
        ok = len(matched) == len(token_variants)
    return {"ok": ok, "matched": matched, "missing": missing, "match_count": len(matched)}

@app.get("/search/filters", summary="Filter by disease and ingredients (ecommerce-like)")
def filtered_search(disease: Optional[str] = None, ingredients: Optional[str] = None, mode: str = "all", limit: int = 20, offset: int = 0):
    """
        Refined search behavior:
        - If only disease provided: return matched diseases and the remedies referenced by those diseases.
        - If only ingredients provided: return diseases whose remedy text mentions those ingredients (match mode rules) AND remedies whose preparation uses them.
        - If both provided: return disease matches; remedies linked to those diseases; remedies using ingredients (independent lists, no intersection section).
        Parameters:
            disease: substring match on disease/name field (case-insensitive)
            ingredients: comma/space separated tokens; Tamil synonyms expanded (milagu→pepper, etc.)
            mode: 'all' (default) all tokens must appear; 'any' at least one
        Response fields:
            - disease_matches
            - diseases_for_ingredients (only populated when disease not given & ingredients provided)
            - remedies_for_disease
            - remedies_using_ingredients (+ ingredient_match_info)
    """
    try:
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        mode = (mode or 'all').lower()
        if mode not in {"all", "any"}:
            mode = "all"

        all_diseases: List[Dict[str, Any]] = getattr(app.state, 'all_diseases', []) or []
        remedies_list: List[Dict[str, Any]] = getattr(app.state, 'remedies_list', []) or []

        # 1) Disease filtering with scoring
        disease_q = (disease or '').strip().lower()
        disease_matches: List[Dict[str, Any]] = []
        if disease_q:
            # Only match against disease/name field, ignore symptoms & remedy text for this mode
            q_tokens = [t for t in disease_q.replace('-', ' ').split() if t]
            for row in all_diseases:
                name = str(row.get('disease') or row.get('name') or '').strip()
                name_l = name.lower()
                if not name_l:
                    continue
                # presence: require every token to appear in order? For now: all tokens contained (substring)
                if not all(tok in name_l for tok in q_tokens):
                    continue
                # scoring: exact > prefix > token coverage
                coverage = sum(1 for tok in q_tokens if tok in name_l) / max(1, len(q_tokens))
                exact = 1.0 if name_l == disease_q else 0.0
                prefix = 1.0 if name_l.startswith(disease_q) and not exact else 0.0
                score = exact * 3 + prefix * 1.5 + coverage
                annotated = dict(row)
                annotated['score'] = round(score,4)
                annotated['token_coverage'] = round(coverage,4)
                disease_matches.append(annotated)
            disease_matches.sort(key=lambda r: (-r.get('score',0), r.get('disease') or r.get('name') or ''))
            # Keep only best single match for disease-only query (UI simplicity per requirement)
            if not ingredients:  # disease-only scenario
                disease_matches = disease_matches[:1]
        disease_matches_page = disease_matches[offset:offset+limit]

        # 2) Ingredient filtering
        ing_tokens = _normalize_ingredient_tokens(ingredients or '')
        token_variants = _expand_token_variants(ing_tokens) if ing_tokens else {}
        remedies_using_ingredients: List[Dict[str, Any]] = []
        ingredient_match_info: List[Dict[str, Any]] = []
        total_tokens = len(token_variants)
        if token_variants:
            for r in remedies_list:
                info = _prep_matches_variants(r.get('Preparation',''), token_variants, mode)
                if info["ok"]:
                    coverage = info['match_count'] / max(1, total_tokens)
                    score = coverage  # simple for now; could add additional weighting later
                    annotated_r = dict(r)
                    annotated_r['score'] = round(score,4)
                    remedies_using_ingredients.append(annotated_r)
                    ingredient_match_info.append({
                        "remedy": _get_remedy_name(r),
                        "matched": info["matched"],
                        "missing": info["missing"],
                        "match_count": info["match_count"],
                        "coverage": round(coverage,4),
                        "score": round(score,4)
                    })
            # sort by score desc
            joined = list(zip(remedies_using_ingredients, ingredient_match_info))
            joined.sort(key=lambda x: (-x[0].get('score',0), x[0].get('Remedy Name') or ''))
            remedies_using_ingredients = [j[0] for j in joined]
            ingredient_match_info = [j[1] for j in joined]
        remedies_using_ingredients_page = remedies_using_ingredients[offset:offset+limit]
        ingredient_match_info_page = ingredient_match_info[offset:offset+limit]

        # 3) Remedies linked to matched diseases (by remedy name mention in 'remedy' field)
        remedies_for_disease: List[Dict[str, Any]] = []
        if disease_matches:
            name_map = {}
            for r in remedies_list:
                nm = _get_remedy_name(r).strip()
                if nm:
                    name_map.setdefault(nm.lower(), r)
            seen = set()
            for d in disease_matches:
                rem_field = str(d.get('remedy') or d.get('Remedy') or '').lower()
                if not rem_field:
                    continue
                for nm, rrow in name_map.items():
                    if nm and nm in rem_field and nm not in seen:
                        seen.add(nm)
                        # Annotate remedy name explicitly
                        annotated_r = dict(rrow)
                        annotated_r['Resolved Name'] = _get_remedy_name(rrow)
                        remedies_for_disease.append(annotated_r)
            # If this is a disease-only query (no ingredients) and exactly one disease, embed remedies directly
            if disease_q and not token_variants and len(disease_matches) == 1:
                disease_matches[0]['primary'] = True
                disease_matches[0]['remedies'] = [
                    {
                        'name': r.get('Resolved Name') or r.get('Remedy Name') or _get_remedy_name(r),
                        'preparation': r.get('Preparation'),
                        'usage': r.get('Usage')
                    } for r in remedies_for_disease
                ]
        remedies_for_disease_page = remedies_for_disease[offset:offset+limit]

        # 4) Diseases inferred from ingredients (only when no explicit disease filter supplied)
        diseases_for_ingredients: List[Dict[str, Any]] = []
        if not disease_q and token_variants:
            for row in all_diseases:
                text = " ".join(str(row.get(k) or '') for k in (
                    'remedy','Remedy','prepared_medicines','Prepared Medicines', 'external_medicines','External Medicines','others','Others'
                )).lower()
                if not text:
                    continue
                matched_tokens = [base for base, vars_ in token_variants.items() if any(v in text for v in vars_)]
                if not matched_tokens:
                    continue
                if mode == 'all' and len(matched_tokens) != len(token_variants):
                    continue
                coverage = len(matched_tokens) / max(1, len(token_variants))
                annotated = dict(row)
                annotated['matched_ingredients'] = matched_tokens
                annotated['score'] = round(coverage,4)
                diseases_for_ingredients.append(annotated)
                if len(diseases_for_ingredients) >= limit + offset:
                    break
            diseases_for_ingredients.sort(key=lambda r: (-r.get('score',0), r.get('disease') or r.get('name') or ''))
        diseases_for_ingredients_page = diseases_for_ingredients[offset:offset+limit]

        response = {
            "query": {
                "disease": disease_q,
                "ingredients": ing_tokens,
                "mode": mode,
                "limit": limit,
                "offset": offset
            },
            "counts": {
                "disease_matches": len(disease_matches),
                "diseases_for_ingredients": len(diseases_for_ingredients),
                "remedies_using_ingredients": len(remedies_using_ingredients),
                "remedies_for_disease": len(remedies_for_disease)
            },
            "disease_matches": disease_matches_page,
            "diseases_for_ingredients": diseases_for_ingredients_page,
            "remedies_using_ingredients": remedies_using_ingredients_page,
            "ingredient_match_info": ingredient_match_info_page,
            "remedies_for_disease": remedies_for_disease_page
        }
        # Ensure no NaN/Inf values are returned (JSON-safe)
        return _json_safe(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform filtered search: {str(e)}")

@app.post("/chat", summary="Endpoint to chat with the bot")
def chat_with_bot(request: ChatRequest):
    """Enhanced chat endpoint with:
    - Multi-turn context (passes formatted history into prompt)
    - Direct herb/remedy lookup when user requests an individual item
    - Fallback to retrieval QA chain
    Returns structured JSON with optional 'lookup' field when a direct match occurs.
    """
    qa_chain = app.state.qa_chain
    herbs_index = getattr(app.state, 'herbs_index', {})
    remedies_index = getattr(app.state, 'remedies_index', {})

    # Early greeting handling (applies to all modes)
    if _is_greeting(request.question):
        return {"answer": "Hello! Ask me about a disease, herb, or remedy. You can also list ingredients (e.g. 'I only have milagu and inji')."}

    # SIMPLE_MODE or missing qa_chain fallback logic
    if SIMPLE_MODE or not qa_chain:
        question_lower = request.question.lower().strip()
        # Direct herb/remedy lookup
        lookup = _detect_direct_lookup(request.question, herbs_index, remedies_index)
        if lookup.get('type'):
            items = lookup['items']
            if lookup['type'] == 'herb':
                parts = []
                for h in items[:3]:
                    parts.append(
                        f"Herb: {h.get('Herb Name (English)','N/A')}\nTamil: {h.get('Tamil Name','N/A')}\nBotanical: {h.get('Botanical Name','N/A')}\nPart Used: {h.get('Part Used','N/A')}\nKey Uses: {h.get('Key Uses','N/A')}\nForm: {h.get('Form of Use','N/A')}"
                    )
                return {"answer": "\n\n".join(parts), "lookup": {"category": "herb", "query": lookup['matched'], "count": len(items)}, "sources": []}
            else:
                parts = []
                referenced = []
                for r_ in items[:3]:
                    name = r_.get('Remedy Name','N/A')
                    parts.append(
                        f"Remedy: {name}\nPreparation: {r_.get('Preparation','N/A')}\nUsage: {r_.get('Usage','N/A')}"
                    )
                    referenced.append({
                        'name': name,
                        'preparation': r_.get('Preparation','N/A'),
                        'usage': r_.get('Usage','N/A'),
                        'url': f"remedies.html?query={name.replace(' ', '%20')}"
                    })
                return {"answer": "\n\n".join(parts), "lookup": {"category": "remedy", "query": lookup['matched'], "count": len(items)}, "sources": [], "referenced_remedies": referenced}

        # Naive disease search
        records = getattr(app.state, 'raw_disease_rows', []) or getattr(app.state, 'all_diseases', [])
        hits = []
        tokens = [t for t in question_lower.split() if t]
        for row in records:
            combined = " ".join(str(v).lower() for v in row.values())
            if tokens and all(tok in combined for tok in tokens):
                hits.append(row)
                if len(hits) >= 3:
                    break
        if hits:
            lines = []
            for h in hits:
                lines.append(
                    f"Disease: {h.get('disease') or h.get('Disease') or h.get('name')}\nSymptoms: {h.get('sign_and_symptoms') or h.get('Sign and Symptoms')}\nRemedy: {h.get('remedy') or h.get('Remedy')}"
                )
            return {"answer": "\n\n".join(lines), "sources": []}

        return {"answer": "(Simple Mode) Ask about a herb (e.g. Turmeric), a remedy (e.g. Thulasi Kashayam) or describe symptoms.", "sources": []}

    # Ingredient-based follow-up detection
    remedies_list_full = getattr(app.state, 'remedies_list', [])
    ingredient_tokens = _detect_ingredient_query(request.question)
    if ingredient_tokens:
        suggestions = _suggest_remedies_for_ingredients(ingredient_tokens, remedies_list_full)
        if suggestions:
            parts = [f"Using your available ingredients ({', '.join(ingredient_tokens)}), here are possible remedies:"]
            rendered = []
            for r_ in suggestions:
                rendered.append(
                    f"**{r_.get('Remedy Name','N/A')}**\n- Preparation: {r_.get('Preparation','N/A')}\n- Usage: {r_.get('Usage','N/A')}"
                )
            parts.append("\n\n".join(rendered))
            return {"answer": "\n\n".join(parts), "suggested_remedies": [
                {
                    'name': r_.get('Remedy Name','N/A'),
                    'preparation': r_.get('Preparation','N/A'),
                    'usage': r_.get('Usage','N/A'),
                    'url': f"remedies.html?query={r_.get('Remedy Name','').replace(' ', '%20')}"
                } for r_ in suggestions
            ]}
        else:
            return {"answer": f"I understood you have: {', '.join(ingredient_tokens)}. I couldn't find a remedy in the dataset that uses all of them together. Try fewer ingredients or ask about one herb.", "suggested_remedies": []}

    # Try direct herb/remedy lookup
    lookup = _detect_direct_lookup(request.question, herbs_index, remedies_index)
    if lookup.get('type'):
        items = lookup['items']
        if lookup['type'] == 'herb':
            herb_blocks = []
            for h in items[:1]:  # show first clearly
                herb_blocks.append(
                    f"**Herb: {h.get('Herb Name (English)','N/A')}**\n"+
                    f"Tamil: {h.get('Tamil Name','N/A')} | Botanical: {h.get('Botanical Name','N/A')} | Part Used: {h.get('Part Used','N/A')}\n"+
                    f"Key Uses: {h.get('Key Uses','N/A')}\n"+
                    f"Forms: {h.get('Form of Use','N/A')}\n"+
                    "Tip: You can ask 'What remedies use Turmeric?' or list ingredients you have."
                )
            answer = "\n\n".join(herb_blocks)
            return {"answer": answer, "lookup": {"category":"herb","query": lookup['matched'],"count": len(items)}}
        else:
            rem_blocks = []
            referenced = []
            for r_ in items[:3]:
                name = r_.get('Remedy Name','N/A')
                rem_blocks.append(
                    f"**Remedy: {name}**\n- Preparation: {r_.get('Preparation','N/A')}\n- Usage: {r_.get('Usage','N/A')}\n- More: remedies.html?query={name.replace(' ','%20')}"
                )
                referenced.append({
                    'name': name,
                    'preparation': r_.get('Preparation','N/A'),
                    'usage': r_.get('Usage','N/A'),
                    'url': f"remedies.html?query={name.replace(' ', '%20')}"
                })
            answer = "\n\n".join(rem_blocks)
            return {"answer": answer, "lookup": {"category":"remedy","query": lookup['matched'],"count": len(items)}, "referenced_remedies": referenced}

    # Disease detection -> include remedies
    disease_rows = _match_diseases(request.question, getattr(app.state, 'all_diseases', []))
    attached_remedies = []
    if disease_rows:
        attached_remedies = _remedies_from_disease_rows(disease_rows, remedies_index, remedies_list_full)
        disease_summaries = []
        for d in disease_rows:
            disease_summaries.append(
                f"**Disease:** {d.get('disease') or d.get('name')}\nSymptoms: {d.get('sign_and_symptoms','N/A')}"
            )
        remedy_render = []
        for r_ in attached_remedies:
            remedy_render.append(
                f"- {r_.get('Remedy Name','N/A')} | Prep: {r_.get('Preparation','N/A')} | Usage: {r_.get('Usage','N/A')}"
            )
        preface = "\n\n**Related Remedies:**\n" + "\n".join(remedy_render) if remedy_render else "\n\n(No direct remedy mapped in dataset)."
        disease_answer = "\n\n".join(disease_summaries) + preface + "\nYou can say: 'I only have milagu and inji' to filter by your ingredients."
        # If in SIMPLE_MODE return now (no RAG)
        if SIMPLE_MODE or not qa_chain:
            return {"answer": disease_answer, "referenced_remedies": [
                {
                    'name': r_.get('Remedy Name','N/A'),
                    'preparation': r_.get('Preparation','N/A'),
                    'usage': r_.get('Usage','N/A'),
                    'url': f"remedies.html?query={r_.get('Remedy Name','').replace(' ', '%20')}"
                } for r_ in attached_remedies
            ]}
        # Otherwise enrich with RAG below after chain invoke

    # Format history string for prompt injection
    history_str = _format_history(request.chat_history)

    # Invoke chain including chat history context
    result = qa_chain.invoke({
        "input": request.question,
        "chat_history": history_str
    })

    sources = [doc.page_content for doc in result.get("context", [])]
    remedies_list = getattr(app.state, 'remedies_list', [])
    referenced = _collect_referenced_remedies([request.question, result["answer"]], remedies_list)
    answer_text = result["answer"]
    # If we earlier detected disease rows, prepend structured disease + remedies info
    if disease_rows:
        remedy_render = []
        for r_ in attached_remedies:
            remedy_render.append(f"- {r_.get('Remedy Name','N/A')}: {r_.get('Preparation','N/A')}")
        disease_block = []
        for d in disease_rows:
            disease_block.append(f"**Disease:** {d.get('disease') or d.get('name')}\nSymptoms: {d.get('sign_and_symptoms','N/A')}")
        prefix = "\n\n".join(disease_block)
        if remedy_render:
            prefix += "\n\n**Relevant Remedies:**\n" + "\n".join(remedy_render) + "\nSay: 'I only have milagu, inji' to filter by ingredients."
        answer_text = prefix + "\n\n" + answer_text
    if referenced:
        addon_lines = ["\n\n**Referenced Remedies:**"]
        for r_ in referenced:
            addon_lines.append(
                f"- **{r_['name']}**\n  - Preparation: {r_['preparation']}\n  - Usage: {r_['usage']}\n  - More: {r_['url']}"
            )
        answer_text += "".join(addon_lines)
    return {"answer": answer_text, "sources": sources, "referenced_remedies": referenced}

# --- Herbs Endpoints ---
@app.get("/herbs", summary="Get all herbs")
def get_herbs():
    try:
        csv_paths = [
            'herb.csv',
            '../backend/herb.csv',
            './herb.csv'
        ]
        herbs = []
        for path in csv_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        herbs = list(reader)
                        break
            except Exception:
                continue
        if not herbs:
            raise FileNotFoundError("Could not find the herb.csv file in any expected location")
        return {"count": len(herbs), "herbs": herbs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load herbs: {str(e)}")

@app.get("/herbs/search", summary="Search for herbs")
def search_herbs(query: str):
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")
    try:
        all_herbs_result = get_herbs()
        all_herbs = all_herbs_result["herbs"]
        q = query.lower()
        results = []
        for herb in all_herbs:
            name = str(herb.get('Herb Name (English)', '')).lower()
            tamil = str(herb.get('Tamil Name', '')).lower()
            botanical = str(herb.get('Botanical Name', '')).lower()
            part = str(herb.get('Part Used', '')).lower()
            uses = str(herb.get('Key Uses', '')).lower()
            form = str(herb.get('Form of Use', '')).lower()
            if (q in name or q in tamil or q in botanical or q in part or q in uses or q in form):
                results.append(herb)
        return {"count": len(results), "herbs": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search herbs: {str(e)}")


