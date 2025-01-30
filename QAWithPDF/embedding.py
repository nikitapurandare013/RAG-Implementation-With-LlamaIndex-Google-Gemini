from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.settings import Settings

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - Query engine for retrieving results.
    """
    try:
        logging.info("Initializing Gemini Embedding model...")

        # Use Settings instead of ServiceContext
        Settings.llm = model
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Creating VectorStoreIndex from documents...")
        index = VectorStoreIndex.from_documents(document)  # No service_context needed
        index.storage_context.persist()

        logging.info("Converting index to query engine...")
        query_engine = index.as_query_engine()
        return query_engine

    except Exception as e:
        raise customexception(e, sys)