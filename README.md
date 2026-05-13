# Document Intelligence RAG System with Faithfulness Evaluation

A production-ready Retrieval-Augmented Generation (RAG) system for document-based question answering with automated evaluation using RAGAS. The system generates grounded and explainable responses by restricting answers to retrieved document context and providing source-level traceability.
Implemented semantic retrieval, query rewriting, conversation-aware responses, and FastAPI-based APIs, along with faithfulness and answer relevancy evaluation to reduce hallucinations and improve response reliability.

## Features

* **Semantic Retrieval:** High-precision document fetching using ChromaDB and Sentence Transformers.
* **LLM-Powered QA:** Context-constrained generation via HuggingFace Inference API (Qwen 2.5).
* **Automated Evaluation:** Built-in RAGAS metrics to score answer **Faithfulness** and **Relevancy**.
* **Query Rewriting:** Automatically optimizes user prompts for better vector search results.
* **Source Attribution:** Returns exact documents and page numbers used to generate the answer.
* **Production-Ready API:** Fast, async REST endpoints built with FastAPI.

## Architecture Flow

```text
User Query ➔ Query Rewriter (LLM) ➔ Vector Retriever (Chroma) ➔ Context-Constrained LLM ➔ Response + Sources ➔ RAGAS Eval
```
## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI |
| **Vector Database** | ChromaDB |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM (QA & Eval)** | `Qwen/Qwen2.5-7B-Instruct` |
| **Evaluation Framework** | RAGAS |
| **Environment** | Python 3.10+ |


## Project Structure
```text
HealthCareRAG/
├── app/
│   └── api.py              # FastAPI routes and endpoints
├── vectordb/               # Persistent Chroma vector store directory
├── rag.py                  # Core RAG logic and RAGAS evaluation pipeline
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables
└── README.md
```

## Getting Started
### 1. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```
### 2. Configure Environment
Create a .env file in the root directory and add your Hugging Face API token:
```bash
HUGGINGFACEHUB_API_TOKEN=XXXXXXXXXXXXXXXX
```
### 3. Start the Server
Run the FastAPI application locally:
```bash
uvicorn main:app --reload
```
The API will be available at http://127.0.0.1:8000.

## Usage Example
Request:
```bash
curl -X POST [http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask) \
-H "Content-Type: application/json" \
-d '{"question": "What are the common side effects of lisinopril?"}'
```
Response:
```json
{
  "answer": "The common side effects of lisinopril include dry cough, dizziness, and mild hypotension.",
  "sources": [
    { "source": "Hypertension_Treatment_Guidelines.pdf", "page": 14 },
    { "source": "Cardiology_Report_2023.pdf", "page": 2 }
  ],
  "evaluation": {
    "faithfulness": 1.0,
    "answer_relevancy": 0.88
  }
}
```
