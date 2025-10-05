# Siddha Medicine Chatbot & Portal

A comprehensive web application featuring a Siddha medicine chatbot (multi‑turn, CSV‑aware) plus searchable databases of diseases, herbs, and remedies grounded in traditional Siddha medicine.

## Features

- 🤖 **Multi‑Turn Chatbot**: Remembers prior turns via `chat_history`; greets politely without false matches
- 🧪 **Context‑Aware Matching**: Greeting inputs ("hi", "hello") no longer trigger disease matches (e.g. prevents "hi" → "hiccups")
- 💊 **Disease Database**: Categorized diseases by dosha (Kabam, Pitham, Vatham)
- 🌿 **Herbs Database**: Herb details (names, uses, properties)
- 🏥 **Remedies Database**: Traditional Siddha remedies with preparation & usage
- 🔍 **Unified Search Page**: Combine a disease term and ingredient list to get the best matching disease and intersected remedies; or search by only disease or only ingredients
- 🧮 **Scoring & Token Highlights**: Backend ranks diseases/remedies by coverage & match score
- 🧾 **Direct Lookups**: Asking about a single herb/remedy returns precise CSV data (no hallucinations in SIMPLE_MODE)
- 🧰 **SIMPLE_MODE**: Lightweight operation without external AI (Gemini / embeddings) for offline or constrained environments
- 📱 **Responsive Design**: Desktop & mobile friendly

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **AI (optional)**: Google Gemini via LangChain + FAISS (when SIMPLE_MODE disabled)
- **Data**: CSV (diseases, herbs, remedies)
- **Processing**: Pandas; lightweight heuristic scoring
- **Embeddings (optional)**: HuggingFace Sentence Transformers

## Prerequisites

- Python 3.8 or higher
- Git
- Google API Key (for Gemini AI)
- HuggingFace API Token (optional, for embeddings)

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/kabillanta/disease.git
cd disease
```

### 2. Create Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Setup

Create a `.env` file in the root directory:

```bash
# Create .env file
touch .env  # On macOS/Linux
# Or manually create the file on Windows
```

Add the following environment variables to `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

#### Getting API Keys:

1. **Google API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

2. **HuggingFace Token** (Optional):
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Create a new token
   - Copy the token to your `.env` file

### 5. Run the Application

#### Start the Backend Server:

```bash
cd backend
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at: `http://127.0.0.1:8000`

#### Serve the Frontend:

Option 1 - Using Python's built-in server:
```bash
cd frontend
python -m http.server 8080
```

Option 2 - Using Live Server (VS Code Extension):
- Install the "Live Server" extension in VS Code
- Right-click on `index.html` and select "Open with Live Server"

Option 3 - Using any static file server:
- Simply serve the `frontend` folder using your preferred method

The web application will be available at: `http://localhost:8080` (or your chosen port)

## Project Structure

```
disease/
├── backend/
│   ├── app.py (optional entry, main server is server.py)
│   ├── server.py                 # FastAPI application
│   ├── herb.csv                  # Herbs dataset
│   ├── remedy.csv                # Remedies dataset
│   ├── cleaned_chatbot_with_type.csv # Processed disease data (with type)
│   └── cleaned_chatbot_with_type.csv (duplicate also in frontend for UI usage)
├── corpus/
│   ├── chatbot.csv               # Original raw disease data
│   └── cleaned_chatbot.csv       # Cleaned disease data
├── frontend/
│   ├── index.html                # Home
│   ├── chatbot.html              # Chat interface
│   ├── diseases.html             # Disease list
│   ├── herbs.html                # Herb list
│   ├── remedies.html             # Remedy list
│   ├── search.html               # Unified search (disease + ingredients)
│   ├── script.js                 # Shared JS (navigation, general)
│   ├── search.js                 # Search page logic
│   ├── style.css                 # Styles
│   └── cleaned_chatbot_with_type.csv # (frontend copy if needed)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .env                          # Environment variables (not committed)
```

## API Endpoints

### Chat Endpoint
- **POST** `/chat` – Multi-turn chat with Siddha medicine assistant
   Request body:
   ```json
   {
      "question": "What are the symptoms of fever?",
      "chat_history": [["I have a cough", "Consider steam inhalation..."]]
   }
   ```
   Behavior:
   - `chat_history`: list of `[user, assistant]` turn pairs; send all previous turns for context
   - Greeting detection: pure greetings return a short friendly message (no disease heuristic)
   - Direct herb/remedy question returns a `lookup` object for deterministic CSV facts
   - SIMPLE_MODE: skips embeddings / Gemini, relies purely on CSV heuristic search
   - Response fields may include:
      - `answer`: main text
      - `lookup`: `{ "category": "herb|remedy", "query": "turmeric" }` when direct match
      - `sources`: (advanced mode only) retrieval sources (may be empty in SIMPLE_MODE)

## SIMPLE_MODE vs Advanced Mode

SIMPLE_MODE keeps everything lightweight and deterministic.

Enable (Windows PowerShell):
```powershell
$env:SIMPLE_MODE="1"; uvicorn server:app --reload --host 127.0.0.1 --port 8000
```
Enable (macOS / Linux):
```bash
SIMPLE_MODE=1 uvicorn server:app --reload --host 127.0.0.1 --port 8000
```
Disable (return to advanced mode): unset or set to 0 and restart.

In SIMPLE_MODE:
- No embeddings / FAISS index / Gemini model
- Faster startup, lower memory
- Chat limited to CSV heuristics + direct lookups

In Advanced Mode (SIMPLE_MODE unset):
- (Optional) Embeddings & retrieval for richer contextual answers
- Potential use of Gemini for generative responses

If API keys are missing, the server will still run in fallback behavior but without generative enhancements.

Note: References to `offline_chat.py` or backup server files have been removed (no longer included in repo).

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Chatbot says it's unavailable | Missing API key or init failed | Set `GOOGLE_API_KEY`, restart, check logs |
| Want only CSV answers | Need lightweight mode | Set `SIMPLE_MODE=1` |
| Can't install faiss / transformers | Limited environment | Use `SIMPLE_MODE` or `offline_chat.py` |
| Herb/Remedy not found | Name mismatch | Try English, Tamil, or botanical variant |

---
Feel free to switch between modes depending on your environment or demo needs.
   - UI can detect `lookup` to style the response differently.

### Disease Endpoints
- **GET** `/diseases/all` – All diseases
- **GET** `/diseases/kabam` – Kabam dosha diseases
- **GET** `/diseases/pitham` – Pitham dosha diseases
- **GET** `/diseases/vatham` – Vatham dosha diseases
- **GET** `/diseases/search?query=fever` – Search diseases by name (heuristic)

### Remedy Endpoints
- **GET** `/remedies` – All remedies
- **GET** `/remedies/search?query=turmeric` – Search remedies (name / ingredient token match)

### Unified Search Endpoint
- **GET** `/search/filters?disease=<term>&ingredients=<comma+separated+tokens>`

Scenarios:
1. Disease only: returns the single best matching disease (by name token coverage & score) with its remedies (embedded). Remedies are filtered for relevance.
2. Ingredients only: returns `diseases_for_ingredients` (diseases related to those ingredients) and `remedies_using_ingredients` ranked by ingredient coverage.
3. Both disease + ingredients: response centers on a single primary disease plus only remedies that include the provided ingredient tokens (intersection). Includes scoring metadata and matched token lists.

Response Fields (subset may appear depending on scenario):
```json
{
   "query": {"disease": "hiccups", "ingredients": ["ginger", "honey"]},
   "disease_match": {"name": "Hiccups", "score": 0.92, "primary": true, "matched_tokens": ["hiccups"]},
   "embedded_remedies": [
       {"name": "Ginger Honey Mix", "ingredients": "Ginger, Honey", "match_score": 0.87, "matched_ingredients": ["ginger", "honey"]}
   ],
   "diseases_for_ingredients": [...],
   "remedies_using_ingredients": [...]
}
```
All numeric fields are JSON-safe (NaN/inf coerced). Fields not applicable are omitted.

## Usage

### 1. Chatbot
- Open `chatbot.html`
- Enter symptoms, conditions, or direct herb/remedy names
- Casual greetings ("hi", "hello") return polite acknowledgements only
- Disease detection ignores pure greetings (prevents false matches like "hi" → "hiccups")
- Direct single-entity questions produce structured lookup answers
- Multi-turn: always append the last Q/A pair to `chat_history` before sending the next question

### 2. Diseases Database
- Browse diseases by dosha type (Kabam, Pitham, Vatham)
- Search for specific diseases
- View detailed symptoms and remedies

### 3. Herbs Database
- Explore medicinal herbs used in Siddha medicine
- Learn about herb properties and usage
- Search for specific herbs

### 4. Remedies Database
- Access traditional Siddha remedies
- View preparation methods and usage instructions
- Search remedies by name or ingredients

### 5. Unified Search Page (`search.html`)
- Input a disease name, ingredient list, or both
- When both are supplied the page shows the top disease match plus intersected remedies only
- Scores and matched tokens help explain relevance
- Layout uses responsive equal-height cards for consistent spacing

## Troubleshooting

### Common Issues

1. **"Module not found" error**:
   - Make sure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

2. **API connection error**:
   - Check if backend server is running on port 8000
   - Verify CORS settings in server.py

3. **Chatbot not responding**:
   - Verify Google API key in `.env` file
   - Check server logs for errors
   - Ensure internet connection for API calls

4. **Frontend not loading**:
   - Check if frontend server is running
   - Verify file paths in HTML files
   - Check browser console for JavaScript errors

### Port Conflicts:
If ports 8000 or 8080 are already in use:

```bash
# For backend (choose different port)
uvicorn server:app --reload --host 127.0.0.1 --port 8001

# For frontend (choose different port)
python -m http.server 8081
```

Remember to update any hard-coded API base URL in JS files if you change the backend port.

## Development

### Adding New Features:
1. Backend changes go in `backend/server.py`
2. Frontend changes go in respective HTML/CSS/JS files
3. New data should be added to CSV files in appropriate directories

### Database Updates:
- `corpus/cleaned_chatbot.csv` - Disease and treatment data
- `backend/herb.csv` - Herbs information
- `backend/remedy.csv` - Remedies information

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/<short-name>`
3. Commit with conventional style if possible: `feat: add fallback remedy matching`
4. Push: `git push origin feature/<short-name>`
5. Open a Pull Request with description & screenshots (if UI)

## License

This project is for educational purposes and follows traditional Siddha medicine practices. Always consult qualified healthcare practitioners for medical advice.

## Disclaimer

This application provides information about traditional Siddha medicine practices for educational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare practitioners for any medical conditions or concerns.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server and browser console logs
3. Create an issue on the GitHub repository
