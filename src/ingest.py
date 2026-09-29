# ingest.py
#
# Ingestion pipeline: load source documents from data/documents/, split them
# into chunks, generate embeddings, and persist them to the local vector store
# in data/vectorstore/.
