# OCR-Document-Summarizer
AI-powered document processing system that extracts text from PDFs and images using OCR and generates concise, context-aware summaries using a BART Transformer model, with an interactive Streamlit interface.
# 📄 AI-Powered Document OCR & Summarization

An AI-powered document processing application that extracts text from PDFs and images using OCR and generates a meaningful, context-aware summary using a pre-trained BART Transformer model.

The application is built using Python and Streamlit and supports both digital and scanned PDF documents.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🖼️ Upload JPG, JPEG, and PNG images
- 🔍 Extract text from documents using OCR
- 📑 Extract text directly from digital PDFs
- 📷 Perform OCR on scanned PDFs
- 🧹 Image preprocessing using OpenCV
- 🤖 AI-powered abstractive text summarization
- 📝 Display extracted text
- 📌 Generate concise and meaningful summaries
- 🌐 Interactive Streamlit web interface

---

## 🏗️ Project Workflow

```text
                PDF / IMAGE
                     │
                     ▼
              Streamlit UI
                     │
                     ▼
              File Detection
               /          \
              /            \
     Digital PDF        Scanned PDF
          │                  │
          ▼                  ▼
      PyMuPDF          Convert to Image
          │                  │
          │                  ▼
          │             OpenCV Processing
          │                  │
          │                  ▼
          │             Tesseract OCR
          │                  │
          └─────────┬────────┘
                    ▼
              Extracted Text
                    │
                    ▼
              Text Cleaning
                    │
                    ▼
             BART Transformer
                    │
                    ▼
            AI-Generated Summary
                    │
                    ▼
              Streamlit UI
