import streamlit as st
import os
from mistralai import Mistral
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
openai_key = os.getenv("api_key_latest")
mistral_key = os.getenv("api_key_mistral")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pages" not in st.session_state:
    st.session_state.pages = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# OCR Function (Runs once per document)
def extract_text_from_mistral(path):
    client = Mistral(api_key=mistral_key)
    uploaded_pdf = client.files.upload(
        file={"file_name": path, "content": open(path, "rb")}, purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        include_image_base64=True
    )
    pages = [page.markdown for page in ocr_response.pages]
    return pages

# Chunk text with page numbers
def chunk_text_with_pages(pages, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " "]
    )
    chunks_with_pages = []
    for page_number, text in enumerate(pages, start=1):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_number))
    return chunks_with_pages

# Create FAISS index once per document
@st.cache_resource
def create_faiss_index_from_text(chunks_with_pages):
    documents = [Document(page_content=chunk, metadata={"page_number": page}) for chunk, page in chunks_with_pages]
    embeddings = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-large")
    faiss_index = FAISS.from_documents(documents, embeddings)
    return faiss_index

# Get RAG response
def get_rag_response(query, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_key, model="gpt-4o-2024-11-20", temperature=0.001),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    sources = set(doc.metadata.get("page_number", "Unknown") for doc in result.get("source_documents", []))
    return result.get("result", "No answer found."), sorted(sources)

# Streamlit UI
st.title("📚 Chat with Your Document via RAG")

# File selection
files = [None] + os.listdir("Document/ArabicDocs")
uploaded_file = st.selectbox("Select a document", files)
path = f"Document/ArabicDocs/{uploaded_file}" if uploaded_file else None

# Process new document
if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
    st.cache_resource.clear()
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.session_state.pages = None
    st.session_state.last_uploaded_file = uploaded_file

# Extract and index document only once
if uploaded_file:
    if st.session_state.pages is None:
        with st.spinner("Extracting text from document..."):
            st.session_state.pages = extract_text_from_mistral(path)
        st.success("Document processed successfully!")

    if st.session_state.vector_store is None:
        with st.spinner("Indexing document..."):
            chunks_with_pages = chunk_text_with_pages(st.session_state.pages)
            st.session_state.vector_store = create_faiss_index_from_text(chunks_with_pages)
        st.success("Document indexed successfully!")

# Chat Interface
if st.session_state.vector_store:
    user_input = st.text_input("Ask a question based on the uploaded document:")
    ask_button = st.button("Ask")

    if ask_button and user_input:
        with st.spinner("Searching..."):
            answer, sources = get_rag_response(user_input, st.session_state.vector_store)
            st.session_state.chat_history.append((user_input, answer, sources))

    # Display Chat History
    if st.session_state.chat_history:
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {question}")
            st.markdown(f"**A{i+1}:** {answer}")
            st.markdown(f"**Source Pages:** {', '.join(map(str, sources)) if sources else 'No sources found.'}")
            st.markdown("---")
else:
    st.info("Please upload a document to start chatting!")
