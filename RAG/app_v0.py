import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import fitz  
from dotenv import load_dotenv
import os
import pytesseract
from pdf2image import convert_from_bytes,convert_from_path
from io import BytesIO

# Load API Key from .env
load_dotenv()
openai_api_key = os.getenv("api_key_latest")
#st.write(openai_api_key)
if not openai_api_key:
    st.error("Please provide a valid OpenAI API key in the .env file.")
    st.stop()


# Streamlit UI
st.title("📚 Chat with Your Document via RAG")
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

# Function to extract Text from scanned PDF
def extract_text_from_scanned_pdf(uploaded_file):

    if uploaded_file is not None:
        images = convert_from_bytes(uploaded_file.getvalue())  # Convert PDF to images
        extracted_text = "\n".join([pytesseract.image_to_string(img,lang="ara") for img in images])
        return extracted_text.strip()

    
# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Function to Extract Text from TXT
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Function to Chunk Text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_text(text)

# Create FAISS Index from Chunked Text
@st.cache_resource
def create_faiss_index_from_text(text):
    # Split text into chunks
    chunks = chunk_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # 🔄 Pass API Key to OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    faiss_index = FAISS.from_documents(documents, embeddings)
    return faiss_index

# RAG Pipeline
def get_rag_response(query, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_api_key, model="gpt-4o-2024-11-20", temperature=0.001),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    return result["result"]

# 🟢 Clear Cache if New File is Uploaded
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
    st.cache_resource.clear()  # Clear the cached FAISS index
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.vector_store = None  # Reset vector store
    st.session_state.chat_history = []    # Reset chat history

# Handle File Upload
if uploaded_file:
    # 🗂️ Extract Text Based on File Type
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
        if(len(document_text)==0):
            document_text = extract_text_from_scanned_pdf(uploaded_file)

    elif uploaded_file.type == "text/plain":
        document_text = extract_text_from_txt(uploaded_file)
        if(len(document_text)==0):
            document_text = extract_text_from_scanned_pdf(uploaded_file)

    st.write(document_text)
            
    # Create FAISS Index and Store in Session
    if st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            st.session_state.vector_store = create_faiss_index_from_text(document_text)
        st.success("Document processed successfully!")

# Chat Interface
if "vector_store" in st.session_state and st.session_state.vector_store:
    user_input = st.text_input("Ask a question based on the uploaded document:")
    ask_button = st.button("Ask")  # 🔘 Ask Button to Trigger RAG

    if ask_button and user_input:  # Trigger only when button is clicked
        with st.spinner("Searching..."):
            answer = get_rag_response(user_input, st.session_state.vector_store)
            # 🗣️ Store question and answer in session state
            st.session_state.chat_history.append((user_input, answer))
    
    # Display Chat History
    if "chat_history" in st.session_state:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {question}")
            st.markdown(f"**A{i+1}:** {answer}")
            st.markdown("---")
else:
    st.info("Please upload a document to start chatting!")
