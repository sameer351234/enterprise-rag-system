 
# Multimodal Enterprise RAG System & AI Agent Deployment

A robust, enterprise-grade Retrieval-Augmented Generation (RAG) system built with Python, LangChain, FAISS, FastAPI, and Streamlit. This application processes multi-format documents (PDFs, TXT, CSVs), indexes text into vector embeddings, and provides precise context-aware answers through an interactive Streamlit UI and FastAPI backend endpoints.

---

## 🌟 Key Features

* **Multi-Format Document Ingestion:** Parses and splits PDFs, text files, and structured CSVs into manageable chunks.
* **Fast Vector Search (FAISS):** Embeds text chunks using HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) and indexes them locally for high-speed semantic retrieval.
* **Context-Aware Retrieval:** Eliminates model hallucinations by providing factual, document-grounded answers.
* **REST API Backend:** Lightweight FastAPI server exposing endpoints for query processing and context retrieval.
* **Interactive UI:** User-friendly Streamlit frontend dashboard displaying answers and source chunk verification.
* **Containerized Deployment:** Fully dockerized for reliable execution across any environment.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **GenAI / Frameworks:** LangChain, HuggingFace Embeddings
* **Vector Database:** FAISS
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit
* **Deployment / Version Control:** Docker, Git, GitHub

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/sameer351234/enterprise-rag-system.git](https://github.com/sameer351234/enterprise-rag-system.git)
cd enterprise-rag-system