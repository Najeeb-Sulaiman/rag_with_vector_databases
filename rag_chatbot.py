# Full Code Implementation for the Q&A app below:

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
#from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore 
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from pinecone import ServerlessSpec, Pinecone
import streamlit as st
from langchain.chat_models import ChatOpenAI
import time
from dotenv import load_dotenv
load_dotenv()


# Load and Process Data
def load_cleaned_data(csv_path):
    """Load cleaned WhatsApp data from a CSV file."""
    data = pd.read_csv(csv_path)
    data['content'] = data['sender'].astype(str) + ': ' + data['message'].astype(str)
    return data['content'].tolist()[4:6]

# Generate Embeddings and Store in Vector Database
def create_vector_store(messages, api_key, index_name):
    """Create a vector database from the messages."""
    # configure client  
    pc = Pinecone(api_key=api_key)  

    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
    
    # check for and delete index if already exists 
    # if index_name in pc.list_indexes().names():  
    #     pc.delete_index(index_name)

    index = pc.Index(index_name)
        
    # Generate embeddings
    model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Check if the index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name, 
            dimension=384, # Dimension for SentenceTransformer embeddings
            metric='cosine',
            spec=spec)
    
        # wait for index to be initialized  
        while not pc.describe_index(index_name).status['ready']:  
            time.sleep(1)  

        # Add data to the vector store
        vector_store = PineconeVectorStore(index, embedding=model, text_key="sentence-transformer")
        for i, embedding in enumerate(messages):
            vector_store.add_texts([messages[i]], metadatas=[{"id": i}])

    else:
        vector_store = PineconeVectorStore(index, embedding=model, text_key="sentence-transformer")

    return vector_store

# RAG Workflow Implementation
def create_rag_chain(vector_store, openai_api_key):
    """Create a Retrieval-Augmented Generation chain using LangChain."""
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-large", 
    #     model_kwargs={"temperature": 0, "max_new_tokens": 200},
    #     huggingfacehub_api_token="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")
    # retriever = vector_store.as_retriever()
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm, 
    #     chain_type="stuff",
    #     retriever=retriever)

    # chatbot language model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-4o-mini',
        temperature=0.0
    )
    # retrieval augmented pipeline for chatbot
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return qa_chain

# Streamlit Web App
def chatbot_ui(qa_chain):
    """Build a simple Streamlit UI for the chatbot."""
    st.title("Data Engineering Community Chatbot")

    user_question = st.text_input("Ask me anything:")
    if user_question:
        response = qa_chain.run(user_question)
        st.write("Answer:", response)

# Main Function to Run the App
def main():
    """Main function to execute the project pipeline."""
    # Load data
    data_path = "messages.csv"  # Replace with your CSV path
    messages = load_cleaned_data(data_path)

    # API Keys (Replace with actual keys or use environment variables)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key=os.getenv("OPENAI_API_KEY")

    index_name = "dec-chat-index"

    # Vector Store
    vector_store = create_vector_store(messages, pinecone_api_key, index_name)

    # RAG Chain
    qa_chain = create_rag_chain(vector_store, openai_api_key)

    # Run Streamlit UI
    chatbot_ui(qa_chain)

if __name__ == "__main__":
    main()
