# Siddha Medicine Chatbot & Portal

A comprehensive web application featuring a Siddha medicine chatbot powered by AI, along with databases of diseases, herbs, and remedies based on traditional Siddha medicine practices.

## Features

- 🤖 **AI Chatbot**: Interactive chatbot for Siddha medicine queries using Google's Gemini AI
- 💊 **Disease Database**: Categorized diseases by dosha types (Kabam, Pitham, Vatham)
- 🌿 **Herbs Database**: Comprehensive herb information with usage and properties
- 🏥 **Remedies Database**: Traditional Siddha remedies and preparations
- 🔍 **Search Functionality**: Search across diseases, herbs, and remedies
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI**: Google Gemini AI via LangChain
- **Vector Database**: FAISS
- **Data Processing**: Pandas
- **Embeddings**: HuggingFace Sentence Transformers

## Prerequisites

- Python 3.8 or higher
- Git
- Google API Key (for Gemini AI)
- HuggingFace API Token (optional, for embeddings)

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/kabillanta/disease-chatbot.git
cd disease-chatbot
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
disease-chatbot/
├── backend/
│   ├── server.py           # FastAPI server
│   ├── herb.csv           # Herbs database
│   └── remedy.csv         # Remedies database
├── corpus/
│   ├── chatbot.csv        # Original disease data
│   └── cleaned_chatbot.csv # Processed disease data
├── frontend/
│   ├── index.html         # Home page
│   ├── chatbot.html       # Chatbot interface
│   ├── diseases.html      # Diseases database
│   ├── herbs.html         # Herbs database
│   ├── remedies.html      # Remedies database
│   ├── script.js          # JavaScript functionality
│   └── style.css          # Styling
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## API Endpoints

### Chat Endpoint
- **POST** `/chat` - Chat with the Siddha medicine bot
  ```json
  {
    "question": "What are the symptoms of fever?",
    "chat_history": []
  }
  ```

### Disease Endpoints
- **GET** `/diseases/all` - Get all diseases
- **GET** `/diseases/kabam` - Get Kabam dosha diseases
- **GET** `/diseases/pitham` - Get Pitham dosha diseases
- **GET** `/diseases/vatham` - Get Vatham dosha diseases
- **GET** `/diseases/search?query=fever` - Search diseases

### Remedy Endpoints
- **GET** `/remedies` - Get all remedies
- **GET** `/remedies/search?query=turmeric` - Search remedies

## Usage

### 1. Chatbot
- Navigate to the chatbot page
- Type your symptoms or health questions
- Get AI-powered responses based on Siddha medicine knowledge

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

## Troubleshooting

### Common Issues:

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

Remember to update the API endpoint in `script.js` if you change the backend port.

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
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Create a Pull Request

## License

This project is for educational purposes and follows traditional Siddha medicine practices. Always consult qualified healthcare practitioners for medical advice.

## Disclaimer

This application provides information about traditional Siddha medicine practices for educational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare practitioners for any medical conditions or concerns.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server and browser console logs
3. Create an issue on the GitHub repository
