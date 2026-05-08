# Market Sentiment Analyzer

AI-powered Market Sentiment Analysis Pipeline using:

- LangGraph
- Google Gemini
- MLflow
- Yahoo Finance
- DuckDuckGo News Search

This project analyzes stock market sentiment for a company by:

1. Resolving stock ticker symbols
2. Fetching latest news
3. Fetching stock market data
4. Running AI-based financial sentiment analysis
5. Generating structured investment insights

---

# Features

- AI-powered sentiment analysis
- Structured JSON output
- Stock price tracking
- News summarization
- Investment recommendations
- MLflow experiment tracking
- LangGraph workflow orchestration

---

# Tech Stack

- Python
- LangGraph
- LangChain
- Google Gemini API
- MLflow
- yFinance
- DuckDuckGo Search

---

# Project Structure

```bash
.
├── market_sentiment_pipeline.py
├── requirements.txt
├── README.md
└── mlflow.db
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/khalisha-sheik/genai-architect-assignments.git
```

---

## 2. Navigate to Project

```bash
cd genai-architect-assignments
```

---

## 3. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Configure GOOGLE_API_KEY

This project uses Google Gemini API.

## Step 1: Generate API Key

Open:

https://aistudio.google.com/app/apikey

Create a new API key.

---

## Step 2: Set Environment Variable

### Mac/Linux

```bash
export GOOGLE_API_KEY="your_google_api_key"
```

### Windows CMD

```cmd
set GOOGLE_API_KEY=your_google_api_key
```

### Windows PowerShell

```powershell
$env:GOOGLE_API_KEY="your_google_api_key"
```

---

# Run the Project

```bash
python market_sentiment_pipeline.py
```

---

# Expected Output

```json
{
  "company_name": "Google",
  "stock_code": "GOOGL",
  "current_price": 172.11,
  "sentiment": "Positive",
  "confidence_score": 0.91,
  "investment_recommendation": "Buy"
}
```

---

# MLflow Tracking

This project uses MLflow for experiment tracking.

To launch MLflow UI:

```bash
mlflow ui
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# Required Python Version

Recommended:

```text
Python 3.10+
```

---

# Install Requirements Example

Example `requirements.txt`:

```txt
mlflow
yfinance
langgraph
langchain
langchain-google-genai
duckduckgo-search
pydantic
```

---

# Future Improvements

- Real-time financial news APIs
- Advanced technical indicators
- Multi-stock comparison
- Dashboard UI
- Vector database integration
- RAG-based financial analysis

---

# Author

Khalisha Sheik
