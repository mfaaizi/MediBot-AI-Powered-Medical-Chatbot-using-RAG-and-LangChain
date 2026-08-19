<div align="center">

#  MediBot
### AI-Powered Medical Chatbot using Retrieval-Augmented Generation (RAG)

*Delivering context-aware and reliable medical information using LangChain, FAISS, and Hugging Face.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-LLM-yellow?style=for-the-badge&logo=huggingface)

</div>

---

#  Overview

**MediBot** is an AI-powered medical question-answering chatbot built using **Retrieval-Augmented Generation (RAG)**. Instead of relying solely on a language model's internal knowledge, MediBot retrieves relevant information from a medical knowledge base before generating responses, resulting in more accurate, reliable, and context-aware answers.

The application combines **LangChain**, **FAISS**, **Sentence Transformers**, and **Hugging Face LLMs** to create an intelligent medical assistant capable of answering domain-specific questions through semantic search and retrieval.

> **Note:** This project is intended for educational and research purposes only and should not be used as a substitute for professional medical advice.

---

#  Features

-  AI-powered medical question answering
-  Retrieval-Augmented Generation (RAG)
-  Semantic document retrieval using FAISS
-  Hugging Face language model integration
-  SentenceTransformer embeddings
-  LangChain RetrievalQA pipeline
-  Custom prompt engineering for accurate responses
-  Clean and interactive Streamlit interface
-  Supports custom medical knowledge bases

---

#  System Architecture

```
Medical Documents
        │
        ▼
Sentence Transformers
        │
        ▼
FAISS Vector Database
        │
        ▼
User Question
        │
        ▼
LangChain RetrievalQA
        │
        ▼
Relevant Context Retrieved
        │
        ▼
Hugging Face LLM
        │
        ▼
Final AI Response
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Embedding Model | SentenceTransformers (MiniLM) |
| Vector Database | FAISS |
| LLM Provider | Hugging Face |
| NLP | Transformers |
| Environment | Pipenv / pip |

---

#  Project Structure

```text
Medibot/
│
├── medibot.py
├── connect_memory_with_llm.py
├── create_memory_for_llm.py
│
├── vectorstore/
│
├── data/
│
├── Pipfile
├── Pipfile.lock
│
└── AI-Medical-Chatbot-with-RAG.pptx
```

---

#  How It Works

### 1. Document Processing

Medical documents are converted into vector embeddings using Sentence Transformers.

### 2. Vector Storage

Embeddings are stored inside a FAISS vector database for efficient similarity search.

### 3. User Query

The user submits a medical question through the Streamlit interface.

### 4. Retrieval

LangChain retrieves the most relevant medical documents from FAISS.

### 5. Response Generation

The retrieved context is provided to a Hugging Face language model, which generates a grounded and context-aware response.

---

#  Installation

## Clone the Repository

```bash
git clone <repository-url>
cd Medibot
```

---

## Install Dependencies

Using pip

```bash
pip install -r requirements.txt
```

or using Pipenv

```bash
pipenv install
```

---

## Configure Hugging Face

Add your Hugging Face credentials.

Required:

- Hugging Face API Token
- Hugging Face Repository ID

---

## Run the Application

```bash
streamlit run medibot.py
```

The application will launch in your browser.

---

#  Project Workflow

1. Load medical documents.
2. Generate embeddings.
3. Store embeddings in FAISS.
4. Receive user question.
5. Retrieve relevant documents.
6. Pass retrieved context to the LLM.
7. Generate a grounded response.
8. Display the answer in the Streamlit interface.

---

#  Use Cases
- Medical education
- Healthcare research
- Medical document exploration
- Retrieval-Augmented Generation demonstrations
- LangChain learning projects
- Semantic search applications

#  Future Improvements

- Conversation memory
- Voice input
- Voice output
- Medical PDF upload
- Multi-language support
- Source citation
- User authentication
- Chat history
- Docker deployment
- Cloud deployment

---

#  Disclaimer

This project is intended solely for educational and research purposes.

The chatbot is **not a licensed medical professional** and should **not** be used for medical diagnosis, treatment, or emergency healthcare decisions.

Always consult a qualified healthcare provider for professional medical advice.

---

#  Author

**Faaiz Imtiaz**

Artificial Intelligence Engineer

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>
