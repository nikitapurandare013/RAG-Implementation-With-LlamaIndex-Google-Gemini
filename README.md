# Retrieval-Augmented Generation (RAG) with LlamaIndex & Google Gemini

This repository provides a comprehensive guide on building a **Retrieval-Augmented Generation (RAG) system** for **Question and Answer (Q&A) with Documents(PDFs amd txt)** using **LlamaIndex** and **Google Gemini**.

## 🚀 Features
- **LlamaIndex for indexing & retrieval** of documents
- **Google Gemini for generating responses** based on retrieved context
- **Efficient embedding and retrieval pipelines**
- **Streamlit-based frontend for user interaction**

### RAG Flowchart explaining the Question Answering with Documents
![image](https://github.com/user-attachments/assets/d609457e-ec21-4e31-a738-a461d059ab42)


### Install dependencies:
```sh
pip install -r requirement.txt
```
## 🏗️ Setup Guide

### 1️⃣ **Set Up Google Gemini API Key**
Obtain an API key from [Google AI Studio](https://ai.google.dev) and set it in your environment:
```sh
export GOOGLE_API_KEY="your-api-key"
```
(For Windows, use `set` instead of `export`.)

### 2️⃣ **Load Documents & Index Data**
```
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents from a folder
documents = SimpleDirectoryReader("./data").load_data()

# Create an index
index = VectorStoreIndex.from_documents(documents)

# Save index
index.storage_context.persist()
```

### 3️⃣ **Query the RAG System**
```
#Function calling
query_engine=download_gemini_embedding(model,document)
response = query_engine.query(user_question)

#Retrieval of index through vector embeddings 
query_engine = index.as_query_engine()

#Response generation

```

### 4️⃣ **Run Streamlit Frontend**
Run the app with:
```sh
streamlit run StreamlitApp.py
```

🔗 **Contributions & Feedback** are welcome! 🔗
