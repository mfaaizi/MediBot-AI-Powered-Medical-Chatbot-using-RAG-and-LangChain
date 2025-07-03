import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens= 512
    )
    return llm


def main():
    st.title("Ask Medibot!")
    st.markdown("**🩺 Your personal AI medical assistant**", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 25px; font-size: 16px;'>
    💬 <b>How it works:</b><br>
    Ask Medibot any medical question and it will reply with context-based answers retrieved from trusted sources.<br><br>

    🧪 <b>Try asking:</b><br>
    • What is a stroke?<br>
    • Symptoms of asthma?<br>
    • Can cancer be cured?
    </div>
    """, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response.get("result", "")
            source_documents = response.get("source_documents", [])

# ✅ Detect "I don't know"-style answers and suppress sources
            irrelevant_phrases = [
            "not mentioned in the provided context",
            "don't know",
            "not enough information",
            "can't answer"
]

            if any(phrase in result.lower() for phrase in irrelevant_phrases):
                source_documents = []  # Clean irrelevant sources


            # ✅ Source starts on a new line after result
            result_to_show = f"{result}\n\n---\n**📚 Source Documents:**\n"
            for i, doc in enumerate(source_documents, 1):
                result_to_show += f"\n**[{i}]** {doc.metadata.get('source', 'Unknown Source')} — page {doc.metadata.get('page_label', 'N/A')}"

            st.chat_message('assistant').markdown(result_to_show, unsafe_allow_html=True)


            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()