# Full Code Implementation for the Q&A app below:

import time
import pandas as pd
import logging
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from pinecone import ServerlessSpec, Pinecone
import streamlit as st


# Load and Process Data
def load_cleaned_data(csv_path):
    """
    Load cleaned WhatsApp data from a CSV file.

    Parameters:
    csv_path (str): Path to the csv for the cleaned whatsapp data.

    Returns:
    The concatenated list of senders and messages.
    """
    try:
        data = pd.read_csv(csv_path)
        data["content"] = (
            data["sender"].astype(str) + ": " + data["message"].astype(str)
        )
        logging.info("Data loaded successfully")
        return data["content"].tolist()
    except Exception as e:
        logging.error(f"Failed to load data from {csv_path}: {str(e)}")
        raise


# Generating Embeddings and Store in Vector Database.
def create_vector_store(messages, api_key, index_name):
    """
    Create a vector database from the messages.

    Parameters:
    messages (list): List of WhatsApp messages.
    api_key (str): Pinecone API KEY. can be gotten from app.pinecone.io.
    index_name (str): Name of the Pinecone Index to be created, if not exist.

    Returns:
    A langchain vectore store object of the generated index.
    """
    # configuring client.
    pc = Pinecone(api_key=api_key)

    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    index = pc.Index(index_name)

    # Creating embedding model object.
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Checking if the index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=384,  # Dimension for SentenceTransformer embeddings
            metric="cosine",  # Vectors similarity search metric
            spec=spec,
        )

        # waiting for index to be initialized
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        logging.info("Index created successfully")

        # Adding data to the vector store.
        vector_store = PineconeVectorStore(
            index, embedding=model, text_key="sentence-transformer"
        )
        for i, embedding in enumerate(messages):
            vector_store.add_texts([messages[i]], metadatas=[{"id": i}])

    else:
        # If index already exist, use it to create a langchain vectore store object.
        vector_store = PineconeVectorStore(
            index, embedding=model, text_key="sentence-transformer"
        )

    return vector_store


# RAG Workflow Implementation
def create_rag_chain(vector_store, openai_api_key):
    """
    Create a Retrieval-Augmented Generation chain using LangChain.

    Parameters:
    vector_store (obj): Landchain vectore store object.
    openai_api_key (str): OpenAI API KEY. can be gotten from platform.openai.com.

    Returns:
    RAG object chaining the created vector index with LLM
    """

    # chatbot language model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0.0
    )
    # retrieval augmented pipeline for chatbot
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )

    return qa_chain


# Streamlit Web App
def chatbot_ui(qa_chain):
    """
    Build a simple Streamlit UI for the chatbot.

    Parameters:
    qa_chain (obj): Object chaining the created vector index with LLM.

    Performs:
    Creates a simple Q&A ui
    """
    st.set_page_config(page_title="Data Engineering Community", page_icon=":bar_chart:")
    st.title("Data Engineering Community Chatbot")
    st.sidebar.image("img/dec_logo.png")

    user_question = st.text_input("Ask me anything:")
    if user_question:
        response = qa_chain.run(user_question)
        st.write("Answer:", response)


# Main Function to Run the App
def main():
    """Main function to execute the project pipeline."""
    # Load data
    data_path = st.secrets["CSV_PATH"]
    messages = load_cleaned_data(data_path)

    # API Keys
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    index_name = "dec-chat"

    # Vector Store
    vector_store = create_vector_store(messages, pinecone_api_key, index_name)

    # RAG Chain
    qa_chain = create_rag_chain(vector_store, openai_api_key)

    # Run Streamlit UI
    chatbot_ui(qa_chain)


if __name__ == "__main__":
    main()
