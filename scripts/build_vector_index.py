import logging
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.helpers.rag import rag_service

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Initializing RAG Index Builder...")
    # Force rebuild
    rag_service.build_index()
    print("Done.")
