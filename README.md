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
- [License](#license)

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


