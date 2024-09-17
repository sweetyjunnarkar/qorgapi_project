import os  # Importing the os module to interact with the operating system  
import streamlit as st  # Importing Streamlit for creating the web application  
from langchain_groq import ChatGroq  # Importing the ChatGroq model for language processing  
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Importing a tool for splitting text into chunks  
from langchain.chains.combine_documents import create_stuff_documents_chain  # Importing a chain for document combination  
from langchain_core.prompts import ChatPromptTemplate  # Importing a template for prompts  
from langchain_community.vectorstores import FAISS  # Importing vector store for efficient document retrieval  
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing loader for PDF files  
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Importing embeddings generator using Google Generative AI  
from dotenv import load_dotenv  # Importing dotenv to load environment variables from a .env file  
from langchain.chains import create_retrieval_chain  # Importing the functionality to create retrieval chains  

# Load environment variables from .env file  
load_dotenv()  

# Retrieve API keys from the environment  
groq_key = os.getenv("QORG_API_KEY")  
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  

# Set the title of the Streamlit app  
st.title("Gemma Model Document Q&A")  

# Initialize the ChatGroq model with the API key and model name  
llm = ChatGroq(api_key=groq_key, model_name="Gemma-7b-it")  

# Create a prompt template for generating responses based on user input and context  
prompt = ChatPromptTemplate.from_template(  
    """  
    Answer the question based on the provided context only.  
    <context>  
    {context}  
    <context>  
    Question:{input}  
    """  
)  

# Function to embed documents into a vector space, aimed at creating a searchable vector store  
def vector_embed():  
    # Check if the 'vectors' key is not set in session state (indicating vectors are not created yet)  
    if "vectors" not in st.session_state:  
        # Initialize embeddings using Google Generative AI model  
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
        # Load PDF documents from the specified directory  
        st.session_state.loader = PyPDFDirectoryLoader("./books")  # Data ingestion from PDF directory  
        st.session_state.docs = st.session_state.loader.load()  # Document loading  
        # Split the documents into manageable text chunks  
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
        # Finalize the splitting process  
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  
        # Create a FAISS vector store from the processed documents and their embeddings  
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  

# User input field for asking questions about the loaded documents  
prompt1 = st.text_input("ask the document")  

# Button to initiate the creation of the vector store  
if st.button("creating vector store"):  
    vector_embed()  # Call the vector embedding function  
    st.write("Vector Db is ready")  # Inform the user that the vector database is prepared  

import time  # Import time module to track processing time  
if prompt1:  # Check if the user has entered a question  
    # Create a document processing chain using the language model and prompt template  
    document_chain = create_stuff_documents_chain(llm, prompt)  
    # Create a retriever from the vector store  
    retriever = st.session_state.vectors.as_retriever()  
    # Generate a retrieval chain that links the retriever to the document processing chain  
    retriever_chain = create_retrieval_chain(retriever, document_chain)  
    start = time.process_time()  # Start timing the response generation  
    response = retriever_chain.invoke({"input": prompt1})  # Invoke the chain to get a response  
    st.write(response['answer'])  # Display the generated answer to the user  

    # Expandable section to show documents similar to the context of the answer provided  
    with st.expander("Document Similarity Search"):  
        for i, doc in enumerate(response['context']):  # Iterate over similar documents  
            st.write(doc.page_content)  # Display the content of each document  
            st.write("---------------------------------------")  # Separator for clarity