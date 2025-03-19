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
from pdf2image import convert_from_bytes
from io import BytesIO

# Load API Key from .env
load_dotenv()
openai_api_key = os.getenv("api_key_latest")
if not openai_api_key:
    st.error("Please provide a valid OpenAI API key in the .env file.")
    st.stop()

# Streamlit UI
st.title("📚 Chat with Your Document via RAG")
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

# Function to extract Text from scanned PDF
def extract_text_from_scanned_pdf(uploaded_file):
    images = convert_from_bytes(uploaded_file.getvalue())  # Convert PDF to images
    extracted_text_with_pages = []
    
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="ara").strip()
        if text:  # Avoid empty extractions
            extracted_text_with_pages.append((text, i + 1))
    
    return extracted_text_with_pages


# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        pages.append((page.get_text(), i + 1))
    doc.close()
    return pages

# Function to Extract Text from TXT
def extract_text_from_txt(txt_file):
    text = txt_file.read().decode("utf-8")
    return [(text, 1)]

# Function to Chunk Text with Page Numbers
def chunk_text_with_pages(pages, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks_with_pages = []
    for text, page_number in pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_number))
    return chunks_with_pages

# Create FAISS Index from Chunked Text
@st.cache_resource
def create_faiss_index_from_text(chunks_with_pages):
    documents = [Document(page_content=chunk, metadata={"page_number": page}) for chunk, page in chunks_with_pages]
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    faiss_index = FAISS.from_documents(documents, embeddings)
    return faiss_index

# RAG Pipeline with Page Numbers
def get_rag_response(query, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_api_key, model="gpt-4o-2024-11-20", temperature=0.001),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    #st.write("Result:")
    #st.write(result)
    sources = set()
    if "source_documents" in result:
        sources = set(doc.metadata.get("page_number", "Unknown") for doc in result["source_documents"])
    return result.get("result", "No answer found."), sorted(sources)

# 🟢 Clear Cache if New File is Uploaded
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
    st.cache_resource.clear()
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.vector_store = None
    st.session_state.chat_history = []

# Handle File Upload
if uploaded_file:
    # 🗂️ Extract Text Based on File Type
    if uploaded_file.type == "application/pdf":
        pages = extract_text_from_pdf(uploaded_file)
        if all(len(i[0]) == 0 for i in pages):
            pages = extract_text_from_scanned_pdf(uploaded_file)
            
    elif uploaded_file.type == "text/plain":
        pages = extract_text_from_txt(uploaded_file)

    st.write(pages)
    st.write(len(pages))
    # Create FAISS Index and Store in Session
    if st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            chunks_with_pages = chunk_text_with_pages(pages)
            #st.write(chunks_with_pages)
            st.session_state.vector_store = create_faiss_index_from_text(chunks_with_pages)
        st.success("Document processed successfully!")

# Chat Interface
if "vector_store" in st.session_state and st.session_state.vector_store:
    user_input = st.text_input("Ask a question based on the uploaded document:")
    ask_button = st.button("Ask")

    if ask_button and user_input:
        with st.spinner("Searching..."):
            answer, sources = get_rag_response(user_input, st.session_state.vector_store)
            st.session_state.chat_history.append((user_input, answer, sources))

    # Display Chat History
    if "chat_history" in st.session_state:
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {question}")
            st.markdown(f"**A{i+1}:** {answer}")
            st.markdown(f"**Source Pages:** {', '.join(map(str, sources)) if sources else 'No sources found.'}")
            st.markdown("---")
else:
    st.info("Please upload a document to start chatting!")
