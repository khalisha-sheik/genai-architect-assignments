# Personalized Course Recommendation Engine

AI-powered Personalized Course Recommendation System using:

- Google Gemini 2.5 Flash
- Gemini Embeddings
- ChromaDB Vector Database
- LangGraph Workflow
- MLflow Tracing
- Retrieval-Augmented Generation (RAG)

---

# Problem Statement

This project implements a semantic course recommendation engine that:

- Understands user learning interests
- Uses embeddings for semantic similarity
- Retrieves relevant courses from a vector database
- Filters completed courses
- Generates AI-powered recommendation rationale

The system follows a Retrieval-Augmented Generation (RAG) architecture.

---

# Features

- Semantic course recommendations
- Vector similarity search
- ChromaDB integration
- Gemini embedding model support
- AI-generated rationale
- MLflow observability
- LangGraph workflow orchestration
- Structured JSON output
- Completed-course filtering

---

# Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 2.5 Flash |
| Embedding Model | gemini-embedding-001 |
| Vector Database | ChromaDB |
| Workflow | LangGraph |
| Framework | LangChain |
| Observability | MLflow |
| Language | Python 3.10+ |

---

# Project Structure

```bash
assignment2/
│
├── course_recommendation_engine.py
├── assignment2dataset.csv
├── requirements.txt
├── README.md
├── sample_output.json
├── .gitignore
└── chroma_db/
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your-github-repo-url>
```

---

## 2. Navigate to Project

```bash
cd assignment2
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
venv\\Scripts\\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Configure GOOGLE_API_KEY

## Generate API Key

Open:

https://aistudio.google.com/app/apikey

Create a new Gemini API key.

---

## Set Environment Variable

### Mac/Linux

```bash
export GOOGLE_API_KEY="your_api_key"
```

### Windows CMD

```cmd
set GOOGLE_API_KEY=your_api_key
```

### Windows PowerShell

```powershell
$env:GOOGLE_API_KEY="your_api_key"
```

---

# Dataset

Place:

```text
assignment2dataset.csv
```

inside project root folder.

---

# Run Project

```bash
python3 course_recommendation_engine.py
```

---

# Expected Output

```json
{
  "user_query": "My background is in ML fundamentals...",
  "completed_courses": ["C001"],
  "recommendations": [
    {
      "rank": 1,
      "course_id": "C002",
      "title": "Deep Learning with TensorFlow and Keras",
      "similarity_score": 0.89,
      "rationale": "This course builds directly on your ML fundamentals..."
    }
  ],
  "total_recommendations": 5,
  "model_used": "gemini-2.5-flash",
  "embedding_model": "gemini-embedding-001"
}
```

---

# MLflow Tracking

Run MLflow UI:

```bash
mlflow ui
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# Architecture Flow

```text
User Query
    ↓
Generate Query Embedding
    ↓
ChromaDB Similarity Search
    ↓
Retrieve Relevant Courses
    ↓
Filter Completed Courses
    ↓
Gemini Generates Rationale
    ↓
Structured JSON Output
```

---

# Sample User Queries

- "I enjoy data visualization and completed Python basics."
- "I want to learn Kubernetes and CI/CD."
- "I am interested in blockchain and smart contracts."
- "I want to specialize in neural networks and MLOps."

---

# Key Concepts Demonstrated

- Embeddings
- Semantic Search
- Vector Databases
- Retrieval-Augmented Generation (RAG)
- LLM Integration
- Workflow Orchestration
- AI Recommendation Systems
- Observability with MLflow

---

# Future Improvements

- Streamlit UI
- Gradio UI
- Hybrid Search
- Learning Path Generation
- Cross-Encoder Re-ranking
- RAGAS Evaluation Metrics

---

# Author

Khalisha Sheik