# Interactive Document Chat Application (RAG)

## Overview

This project is a sophisticated, interactive chatbot built with Streamlit that allows users to upload multiple documents (PDF, DOCX, TXT) and have a conversation about their content. The application leverages a powerful, open-source Retrieval-Augmented Generation (RAG) pipeline to provide answers that are grounded in the provided text, ensuring accuracy and minimizing hallucinations.

The backend is architected using LangChain and is powered by the high-speed Llama 3 model via the Groq API, making the chat experience incredibly fast and responsive. This project was developed as part of the Generative AI August 2024 Task and represents a robust, production-ready implementation of the core requirements.

## Live Demo

**[Streamlit Demo Link]**(https://interactive-document-chat-application.streamlit.app/)

## Features

-   **Multi-File Upload**: Supports PDF, DOCX, and TXT formats simultaneously.
-   **Interactive Chat Interface**: A clean and intuitive chat window powered by Streamlit.
-   **Robust RAG Pipeline**: Implements a reliable, linear RAG chain using the latest LangChain components (`create_retrieval_chain`) for optimal performance.
-   **View Sources**: Each answer is accompanied by an expandable section showing the exact text chunks from the source documents that were used to generate the response, building user trust.
-   **High-Speed LLM**: Utilizes the `llama3-8b-8192` model via the Groq API for near-instantaneous response times.
-   **Secure & Deployable**: Uses a dual-method for API key management (`.env` for local development, Streamlit Secrets for cloud deployment), ensuring no keys are ever exposed in the repository.
-   **Enhanced User Experience**: Features a multi-step progress bar during document processing to provide clear feedback to the user on slow operations like embedding calculation.

## System Architecture

The application follows a classic and robust RAG architecture, designed for reliability and performance.

```plaintext
+------------------+     +------------------------+      +-------------------+
|  User Uploads    |---->|   Load & Chunk Docs    |----->| Create VectorDB   |
| (PDF, DOCX, TXT) |     | (Recursive Splitter)   |      | (FAISS + MiniLM)  |
+------------------+     +------------------------+      +---------+---------+
                                                                  |
                                                                  | (Vector Store Ready)
                                                                  |
+------------------+     +------------------------+      +--------v---------+
|   User Question  |---->|   Retrieve Relevant    |----->|  Stuff Context   |
|   (Chat Input)   |     |    Document Chunks     |      |   into Prompt    |
+------------------+     +------------------------+      +--------+---------+
                                                                  |
                                                                  |
                                                      +-----------v----------+
                                                      |   Generate Answer    |
                                                      |  (Groq Llama 3 LLM)  |
                                                      +-----------+----------+
                                                                  |
                                                                  V
+------------------+                                  +----------------------+
|  Display Answer  |<----------------------------------|  Answer + Sources   |
|   & Sources      |                                  |   (Final Result)     |
+------------------+                                  +----------------------+

```

1.  **Document Processing**: Uploaded files are loaded, parsed, and split into uniform text chunks.
2.  **Vectorization**: Each chunk is converted into a numerical vector using a `Sentence-Transformers` model and stored in an in-memory FAISS vector database.
3.  **Retrieval**: When a user asks a question, the application retrieves the most relevant document chunks from the vector database.
4.  **Augmentation**: The retrieved chunks are "stuffed" into a prompt along with the user's question.
5.  **Generation**: The complete prompt is sent to the Llama 3 model via the Groq API, which generates a final answer grounded in the provided context.
6.  **Citation**: The final answer and the source chunks are returned to the UI for the user to review.

## Tech Stack

-   **Application Framework**: Streamlit
-   **Core AI Logic**: LangChain
-   **LLM Provider**: Groq (Llama 3 8B)
-   **Embedding Model**: `all-MiniLM-L6-v2` (from Hugging Face)
-   **Vector Store**: FAISS (In-memory)
-   **Document Loaders**: PyPDF, Docx2Txt

## Setup and Installation

Follow these steps to run the application on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AlphaPriyan08/interactive-document-chat-application
    cd interactive-document-chat-application
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a secrets file:**
    -   Create a file named `.env` in the root of your project directory.
    -   Sign up for a free Groq API key at [console.groq.com](https://console.groq.com/).
    -   Add your API key to the `.env` file like this:
        ```
        GROQ_API_KEY="gsk_...your...key...here"
        ```

## How to Run

Once the setup is complete, run the following command in your terminal:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.