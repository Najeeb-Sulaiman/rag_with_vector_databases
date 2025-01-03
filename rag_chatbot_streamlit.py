from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone
import streamlit as st


# Generating Embeddings and Store in Vector Database.
def create_vector_store(api_key, index_name):
    """
    Create a Langchain vector object from the existing index in Pinecone database.

    Parameters:
    api_key (str): Pinecone API KEY. can be gotten from app.pinecone.io.
    index_name (str): Name of the Pinecone Index to be created, if not exist.

    Returns:
    A langchain vectore store object of the index.
    """
    # configuring client.
    pc = Pinecone(api_key=api_key)

    index = pc.Index(index_name)

    # Creating embedding model object.
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    RAG object chaining the vector index with LLM
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
    """Main function to execute the Streamlit app."""

    # API Keys
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    index_name = "dec-chat"

    # Vector Store
    vector_store = create_vector_store(pinecone_api_key, index_name)

    # RAG Chain
    qa_chain = create_rag_chain(vector_store, openai_api_key)

    # Run Streamlit UI
    chatbot_ui(qa_chain)


if __name__ == "__main__":
    main()
