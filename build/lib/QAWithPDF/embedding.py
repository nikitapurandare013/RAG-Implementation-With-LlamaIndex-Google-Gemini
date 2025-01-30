from llama_index.core import VectorStoreIndex, Document
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from custom_exception import customexception
from logger import logging

def download_gemini_embedding(model, documents):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Parameters:
    - model: The language model for LLM operations (if needed for other purposes).
    - documents: List of Document objects to be processed.

    Returns:
    - query_engine: A query engine initialized with the vector store index.
    """
    try:
        logging.info("Initializing the Gemini Embedding model...")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

        logging.info("Splitting documents into chunks...")
        # Initialize the text splitter
        text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=20)

        # Apply text splitting to all documents
        split_documents = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.text)  # Split the text into chunks
            split_documents.extend([Document(text=chunk) for chunk in chunks])  # Wrap chunks into Document objects

        logging.info("Creating the VectorStoreIndex with the embedding model...")
        # Create the VectorStoreIndex with the embedding model and the split documents
        index = VectorStoreIndex.from_documents(
            split_documents,
            embed_model=gemini_embed_model,
            llm=model  # Use the model in the embedding process
        )

        logging.info("Persisting the index to storage...")
        # Persist the index (optional but recommended)
        index.storage_context.persist()

        logging.info("Creating the query engine...")
        # Create and return a query engine from the index
        query_engine = index.as_query_engine()
        return query_engine

    except Exception as e:
        raise customexception(e, sys)
