🩺 MediBot – AI-Powered Medical Chatbot using RAG and LangChain

MediBot is a medical question-answering chatbot that leverages Retrieval-Augmented Generation (RAG) to provide precise, context-aware answers to user queries. It uses LangChain, FAISS for vector storage, Hugging Face for LLM integration, and Streamlit for an intuitive web interface.

🚀 Features

Uses RAG architecture for grounded and accurate responses

Embeds medical documents using HuggingFace Embeddings

Retrieves relevant context via FAISS vector search

Generates responses using a Hugging Face-hosted LLM

Custom prompt engineering to align response tone and clarity

Clean and interactive Streamlit interface

🧠 Tech Stack

Python

Streamlit

LangChain

HuggingFace Transformers & Endpoints

SentenceTransformers (MiniLM)

FAISS (vector database)

📁 Project Structure

graphql
Copy
Edit
Medibot/
├── medibot.py                  # Main Streamlit app

├── connect_memory_with_llm.py # Logic to query vector store

├── create_memory_for_llm.py   # Embeds medical PDFs into FAISS

├── vectorstore/               # Stored FAISS vector index

├── data/                      # Source documents (medical knowledge base)

├── Pipfile & Pipfile.lock     # Project environment

└── AI-Medical-Chatbot-with-RAG.pptx  # Presentation

⚙️ How It Works

Medical documents are embedded into vector space using SentenceTransformers.

User queries are processed via LangChain's RetrievalQA chain.

FAISS retrieves relevant context; the LLM responds based on this evidence.

Responses are streamed through a Streamlit chatbot UI.

▶️ Running the App

Clone the repository

Install dependencies (pipenv or pip):

pip install -r requirements.txt or pipenv install

Run the chatbot:

streamlit run medibot.py

Note: You’ll need a Hugging Face API token and repo ID.

🩻 Use Case

MediBot is designed for educational and informational purposes only. It demonstrates how domain-specific LLMs combined with retrieval techniques can deliver more accurate and trustworthy answers.
