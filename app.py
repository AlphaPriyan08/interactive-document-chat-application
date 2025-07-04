# --------------------------------------------------------------------------
# Document Chat Application (RAG)
#
# Author: Priyadharsan K
# Date: 04.07.2025
#
# Description: An interactive Streamlit application that allows users to
# upload documents and chat with them using a robust RAG pipeline
# powered by LangChain and Groq.
# --------------------------------------------------------------------------

import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("🚨 Groq API Key not found! Please set it in your .env or Streamlit secrets.", icon="🔥")
        st.stop()

LLM_MODEL = "llama3-8b-8192" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UPLOAD_PATH = "uploads"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# --- 2. DOCUMENT PROCESSING ---
@st.cache_resource
def load_and_process_documents(uploaded_files):
    """Loads, chunks, and prepares documents."""
    docs = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_PATH, file.name)
        with open(file_path, "wb") as f: f.write(file.getbuffer())
        loader = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}.get(os.path.splitext(file.name)[1])
        if loader: docs.extend(loader(file_path).load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

@st.cache_resource
def create_vector_store(_texts):
    """Creates the vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(documents=_texts, embedding=embeddings)

# --- 3. THE CORE RAG ENGINE ---
def create_rag_chain(vector_store):
    """
    Builds the core Retrieval-Augmented Generation chain.
    This is a robust, linear chain that is much more reliable.
    """
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.2, api_key=api_key)
  
    prompt = PromptTemplate.from_template(
        """You are an expert assistant for question-answering tasks.
        Use ONLY the following retrieved context to answer the question.
        If you don't know the answer from the provided context, just say that you don't know.
        Your answer should be concise and helpful.

        CONTEXT:
        {context}

        QUESTION: {input}

        ANSWER:"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector_store.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# --- 4. STREAMLIT UI ---
st.title("📄 Document Chat Application")
st.markdown("Upload your documents and ask questions about their content. This application uses a robust RAG pipeline to provide answers grounded in your documents.")

if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)  
               
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0, text="Starting...")
                
                progress_bar.progress(10, text="Loading and chunking...")
                chunked_docs = load_and_process_documents(uploaded_files)
                
                progress_bar.progress(40, text="Creating embeddings (this can be slow)...")
                vector_store = create_vector_store(chunked_docs)
                
                progress_bar.progress(80, text="Building the RAG chain...")
                st.session_state.rag_chain = create_rag_chain(vector_store)
                
                progress_bar.progress(100, text="Done!")
                st.success("Documents processed! You can now ask questions.")
                progress_bar.empty()
                
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()  
                
            
                
st.header("2. Ask Questions")
if st.session_state.rag_chain is None:
    st.info("Please upload and process your documents in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finding answers..."):
                result = st.session_state.rag_chain.invoke({"input": prompt})
                
                response = result["answer"]
                st.markdown(response)

                with st.expander("View Sources"):
                    if result.get("context"):
                        for doc in result["context"]:
                            source_name = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**From:** `{os.path.basename(source_name)}`")
                            st.markdown(f"> {doc.page_content.strip()}")
                            st.markdown("---")
                    else:
                        st.write("No sources were used for this response.")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        