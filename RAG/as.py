import streamlit as st 
import os
from pathlib import Path
from mistralai import Mistral
from IPython.core.display import display, Markdown
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from dotenv import load_dotenv
import re
import tempfile

# Load environment variables
load_dotenv()
openai_key = os.getenv("api_key_latest")
mistral_key = os.getenv("api_key_mistral")


# OCR Function
def mistral_ocr(uploaded_file):
    client = Mistral(api_key=mistral_key)
    
    # Convert the uploaded file to bytes
    file_bytes = uploaded_file.getvalue()
    
    # Upload the file directly using bytes content
    uploaded_pdf = client.files.upload(
        file={"file_name": uploaded_file.name, "content": file_bytes}, purpose="ocr"
    )
    
    # Retrieve file and get signed URL
    client.files.retrieve(file_id=uploaded_pdf.id)
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    
    # Process OCR with the signed URL
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=True
    )

    return ocr_response


# Replace Images in Markdown
def replace_images_in_markdown_for_all_pages(ocr_response):
    for i in range(len(ocr_response.pages)):
        markdown = ocr_response.pages[i].markdown
        images = ocr_response.pages[i].images

        # Create a dictionary mapping image ids to base64 data for the current page
        image_base64_dict = {image.id: image.image_base64 for image in images}

        # Function to replace image references with base64 data in Markdown
        def replace(match):
            img_id = match.group(1)
            base64_data = image_base64_dict.get(img_id)
            if base64_data:
                return f'![{img_id}]({base64_data})'
            return match.group(0)

        # Replace all image references in the current page's markdown
        updated_markdown = re.sub(r'! (.*?) (.*?)', replace, markdown)

        # Display the updated Markdown for the current page
        display(Markdown(updated_markdown))


# Clean base64 string
def clean_base64(base64_str):
    if base64_str.startswith('data:image/jpeg;base64,'):
        return base64_str.replace('data:image/jpeg;base64,', '')
    return base64_str


# Extract text and handle image summaries
def replace_images_with_summary_in_markdown(ocr_response):
    client = Mistral(mistral_key)
    updated_markdown_list = []

    for i in range(len(ocr_response.pages)):
        markdown = ocr_response.pages[i].markdown
        images = ocr_response.pages[i].images

        # Create a dictionary mapping image ids to base64 data for the current page
        image_base64_dict = {image.id: image.image_base64 for image in images}

        # Replace images with summaries
        def replace(match):
            img_id = match.group(1)
            base64_data = image_base64_dict.get(img_id)

            if base64_data:
                cleaned_base64 = clean_base64(base64_data)
                image_summary = image_summarization(client, cleaned_base64)
                return f'[Image Summary: {img_id}] - {image_summary}'

            return match.group(0)

        updated_markdown = re.sub(r'!(.*?)(.*?)', replace, markdown)
        updated_markdown_list.append(updated_markdown)

    final_markdown = []
    for page_num in range(len(updated_markdown_list)):
        final_markdown.append((page_num + 1, updated_markdown_list[page_num]))

    return final_markdown


# Extract text from Mistral OCR response
def extract_text_from_mistral(path):
    #path="/workspaces/AgenticAI-Systems/RAG/Document/مراسلات.pdf"
    ocr_response = mistral_ocr(path)
    replace_images_in_markdown_for_all_pages(ocr_response)
    final_markdown = replace_images_with_summary_in_markdown(ocr_response)
    return final_markdown


# Chunk text with page numbers
def chunk_text_with_pages(pages, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks_with_pages = []
    for page_number, text in pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_number))
    return chunks_with_pages


# Create FAISS index
@st.cache_resource
def create_faiss_index_from_text(chunks_with_pages):
    documents = [Document(page_content=chunk, metadata={"page_number": page}) for chunk, page in chunks_with_pages]
    embeddings = OpenAIEmbeddings(api_key=openai_key)
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
    sources = set()
    if "source_documents" in result:
        sources = set(doc.metadata.get("page_number", "Unknown") for doc in result["source_documents"])
    return result.get("result", "No answer found."), sorted(sources)


# Streamlit UI
st.title("📚 Chat with Your Document via RAG")
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

# Session State Initialization
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
    st.cache_resource.clear()
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.pages = None

# Handle File Upload
if uploaded_file:
    # Extract text once
    if "pages" not in st.session_state or st.session_state.pages is None:
        with st.spinner("Extracting text from document..."):
            if uploaded_file.type in ["application/pdf", "text/plain"]:
                st.session_state.pages = extract_text_from_mistral(uploaded_file)
        st.success("Document processed successfully!")

    st.write(st.session_state.pages)

    # Create FAISS Index once
    with st.spinner("Indexing document..."):
        chunks_with_pages = chunk_text_with_pages(st.session_state.pages)
        st.session_state.vector_store = create_faiss_index_from_text(chunks_with_pages)
    st.success("Document indexed successfully!")

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