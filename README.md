# Video RAG: Retrieval-Augmented Question Answering over YouTube Videos

## Overview

Video RAG is a Retrieval-Augmented Generation (RAG) system designed to answer natural language questions about YouTube videos. Instead of relying solely on a Large Language Model, the system first retrieves the most relevant transcript segments using a hybrid retrieval strategy, then generates grounded answers based on the retrieved context.

The project provides an end-to-end pipeline including transcript extraction, semantic chunking, vector indexing, keyword retrieval, and answer generation through Google's Gemini model. A Streamlit frontend enables users to upload YouTube videos, monitor the indexing process, browse transcripts, and interact with the system through a conversational interface.

---

## Features

* Process YouTube videos automatically
* Retrieve transcripts from multiple sources

  * Human-generated captions
  * Auto-generated captions
  * Whisper transcription (fallback)
* Transcript normalization and cleaning
* Semantic chunking
* Hybrid Retrieval

  * Dense Retrieval using sentence embeddings
  * Sparse Retrieval using BM25
  * Reciprocal Rank Fusion (RRF)
* Grounded answer generation using Gemini
* Timestamp-based evidence
* Interactive transcript viewer
* Jump directly to relevant timestamps in the video
* Suggested questions generated from transcript content
* Asynchronous indexing pipeline with background processing

---

## System Architecture

```
YouTube Video
      │
      ▼
Transcript Source Selection
(Human → Auto → Whisper)
      │
      ▼
Transcript Normalization
      │
      ▼
Semantic Chunking
      │
      ├──────────────┐
      ▼              ▼
Dense Embedding     BM25 Index
(BGE)               (Sparse)
      │              │
      └──────┬───────┘
             ▼
      Reciprocal Rank Fusion
             │
             ▼
      Retrieved Context
             │
             ▼
 Gemini Answer Generation
             │
             ▼
 Answer + Evidence + Timestamp
```

---

# Models Used

| Component                     | Model                                                          |
| ----------------------------- | -------------------------------------------------------------- |
| Dense Embedding               | BAAI/bge-base-en-v1.5 *(or another SentenceTransformer model)* |
| Sparse Retrieval              | BM25                                                           |
| LLM                           | Gemini 3.1 Flash-lite                                          |
| Speech Recognition (Fallback) | Whisper                                                        |
| Transcript Source             | YouTube Transcript API                                         |

---

# Project Structure

```
backend/
│
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── video.py
│   │   │   └── search.py
│   │   └── router.py
│   │
│   ├── core/
│   │   ├── cache.py
│   │   ├── constants.py
│   │   └── jobs.py
│   │
│   ├── services/
│   │   ├── transcript/
│   │   ├── preprocessing/
│   │   ├── vector_store/
│   │   ├── retrieval/
│   │   └── generation/
│   │
│   ├── config.py
│   └── main.py
│
└── storage/
    ├── audio_cache/
    ├── faiss_indices/
    ├── bm25_indices/
    └── response_cache/


frontend/
│
├── app.py
├── requirements.txt
└── components/
    ├── video_player.py
    ├── preview_card.py
    └── search_interface.py
```

---

# Backend Pipeline

### 1. Transcript Collection

The system attempts to retrieve transcripts using the following priority:

1. Human-generated captions
2. Auto-generated captions
3. Whisper transcription (fallback)

---

### 2. Transcript Preprocessing

The transcript is cleaned before indexing:

* remove noise
* remove music tags
* normalize punctuation
* normalize whitespace
* sentence segmentation

---

### 3. Semantic Chunking

Instead of splitting every fixed number of characters, adjacent sentences are grouped according to semantic similarity to preserve contextual meaning.

---

### 4. Hybrid Retrieval

For each user query:

* Dense Retrieval searches semantically similar chunks.
* BM25 retrieves exact keyword matches.
* Reciprocal Rank Fusion combines both rankings into a unified result.

---

### 5. Answer Generation

The retrieved transcript chunks are passed to Gemini together with the user query.

Gemini generates:

* grounded answer
* supporting transcript snippets
* corresponding timestamps

---

# Frontend

The Streamlit interface provides:

* YouTube URL input
* asynchronous indexing status
* embedded YouTube player
* transcript browser
* timestamp navigation
* conversational search
* evidence visualization
* suggested questions

---

# API Endpoints

## Video Processing

```
POST /api/video/process
```

Starts asynchronous indexing.

---

## Pipeline Status

```
GET /api/video/status/{video_id}
```

Returns the current processing stage.

Example:

```json
{
    "status": "chunking"
}
```

---

## Transcript

```
GET /api/video/transcript
```

Returns the complete transcript.

---

## Suggested Questions

```
GET /api/video/suggestions
```

Returns Gemini-generated questions.

---

## Search

```
POST /api/search
```

Input

```json
{
    "video_id":"xxxx",
    "query":"What is semantic chunking?"
}
```

Output

```json
{
    "answer":"...",
    "preview":[
        "...",
        "..."
    ],
    "timestamps":[
        85,
        126
    ]
}
```

---

# Installation

Clone the repository

```bash
git clone <repository_url>
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

---

# Running the Project

## Start Backend

```bash
cd backend

uvicorn app.main:app --reload
```

Backend will be available at

```
http://127.0.0.1:8000
```

---

## Start Frontend

```bash
cd frontend

streamlit run app.py
```

Frontend will be available at

```
http://localhost:8501
```

---

# Example Workflow

1. Paste a YouTube URL.
2. Start the indexing pipeline.
3. Wait until processing is completed.
4. Ask questions about the video.
5. Review retrieved transcript evidence.
6. Click timestamps to jump directly to the corresponding position in the video.

---

# Demo

### Video Processing

```
Input:
https://www.youtube.com/watch?v=xxxxxxxxxxx
```

The backend performs:

* transcript extraction
* transcript cleaning
* semantic chunking
* dense indexing
* BM25 indexing

---

### Question Answering

**User**

```
Explain what semantic chunking is.
```

**System**

```
Semantic chunking groups semantically related sentences into coherent text segments before indexing, preserving contextual information during retrieval.
```

Evidence:

```
[03:45]
Semantic chunking divides text according to semantic similarity rather than fixed length.
```

Clicking the timestamp immediately seeks the embedded YouTube player to **03:45**.

---

# Future Improvements

* Multilingual embedding models
* OCR support for slide-based videos
* Multi-video retrieval
* Vector database integration (Qdrant/Milvus)
* Conversation memory across sessions
* User authentication
* Docker deployment
* Cloud storage support

---

# License

This project is intended for educational and research purposes.
