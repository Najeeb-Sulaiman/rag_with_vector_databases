# GenAI With Vector Databases

This repository contains the code and resources for building a **Retrieval-Augmented Generation (RAG)** chatbot using WhatsApp group chat data. The chatbot is designed to answer questions based on historical messages from the group, showcasing how vector databases and generative AI can be applied in real-world scenarios.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Setup](#project-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Useful Resources](#useful-resources)

---

## Overview
The goal of this project is to demonstrate the power of vector databases and generative AI by building a chatbot that can:
1. Ingest WhatsApp group chat messages.
2. Generate embeddings for the messages and store them in a vector database.
3. Use a Retrieval-Augmented Generation (RAG) pipeline to answer user questions based on the stored messages.

This project is designed to be educational and interactive, suitable for exploration and learning.

---

## Features
- **Custom Data**: Use cleaned WhatsApp group chat data as the knowledge base.
- **RAG Workflow**: Combines vector-based search with a Hugging Face LLM for accurate and context-aware answers.
- **Streamlit UI**: A simple and user-friendly interface for interacting with the chatbot.
- **Scalability**: Demonstrates integration with Pinecone for vector storage.

---

## Technologies Used
- **Python**: Main programming language.
- **Streamlit**: For creating the chatbot's web interface.
- **Hugging Face Transformers**: Provides the LLM for generating responses.
- **Sentence Transformers**: For generating text embeddings.
- **Pinecone**: As the vector database for storing and retrieving embeddings.
- **Pandas**: For data cleaning and manipulation.

---

## Project Setup

### Prerequisites
1. Python 3.8 or higher installed on your system.
2. API keys for Pinecone and Hugging Face Hub.
3. A cleaned CSV file of WhatsApp messages with the following fields:
   - `sender`
   - `message`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Najeeb-Sulaiman/rag_with_vector_databases.git
   cd rag_with_vector_databases
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables
Create a `.env` file in the project root and add the following:
```
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACEHUB_API_TOKEN=your-huggingface-api-token
```
Replace `your-pinecone-api-key`, `your-openai-api-key`, and `your-huggingface-api-token` with your actual credentials.

---

## How It Works

1. **Data Loading**:
   - Load the cleaned WhatsApp chat data from a CSV file.
   - Concatenate `sender` and `message` fields to create context-rich content.

2. **Embedding Generation**:
   - Use `SentenceTransformer` to generate embeddings for each message.

3. **Vector Storage**:
   - Store embeddings in Pinecone, associating each with metadata (e.g., message text).

4. **RAG Workflow**:
   - Retrieve relevant messages using similarity search in Pinecone.
   - Pass the retrieved messages as context to the Hugging Face or OpenAI LLM.

5. **Streamlit UI**:
   - A web interface to input user questions and display chatbot responses.

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run rag_chatbot.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

3. Ask questions based on the WhatsApp chat data and receive intelligent answers!

---

## Future Enhancements
- Add support for other vector databases like Milvus or Weaviate.
- Add file upload option and support Q&A based on uploaded document.
- Expand to handle multimedia data like images and audio.

---

## Useful Resources
[Pinecone documentation](https://docs.pinecone.io/guides/get-started/overview)

[Pinecone Learn](https://www.pinecone.io/learn/)

[Pinecone Example Project Github Repo](https://github.com/pinecone-io/examples)

[Langchain documentation](https://python.langchain.com/docs/introduction/)

[Langchain  Tutorials](https://python.langchain.com/docs/tutorials/)

[LangChain Github Repo](https://github.com/langchain-ai/langchain)

[Hugging face documentation](https://huggingface.co/docs)

[OpenAI API reference](https://platform.openai.com/docs/api-reference/introduction)

[Google ML cash course](https://developers.google.com/machine-learning/intro-to-ml)

[Jay Alammar Word2Vec Illustration](https://jalammar.github.io/illustrated-word2vec/)

