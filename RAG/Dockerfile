# Use the official Python image as a base
FROM python:3.12.1-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-hin tesseract-ocr-fra tesseract-ocr-ara && rm -rf /var/lib/apt/lists/*

# Copy the app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app_mistral_v2.py"]
