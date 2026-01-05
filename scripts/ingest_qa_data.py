import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_stable_id(topic: str, question: str) -> str:
    """Generate a stable 8-char ID based on topic and question content."""
    content = f"{topic}|{question}"
    return hashlib.md5(content.encode()).hexdigest()[:8]

def clean_text(text) -> str:
    """Clean and normalize text fields."""
    if pd.isna(text):
        return ""
    return str(text).strip()

def process_excel(input_path: str, output_path: str):
    """
    Reads Q&A Excel and converts to structured JSON for RAG.
    Schema:
    {
        "id": "stable_hash",
        "topic": "Category",
        "question": "User Query",
        "answer": "Bot Response",
        "metadata": {
            "source_file": "Q&A_pairs.xlsx",
            "tags": []
        }
    }
    """
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading {input_path}...")
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return
    
    # Expected columns: 'pragraph' -> topic, 'question' -> question, 'answer' -> answer
    # Rename for clarity if needed, or just use as is.
    # The user showed columns: ['pragraph', 'question', 'answer', 'Unnamed: 3']
    
    knowledge_base: List[Dict] = []
    
    for idx, row in df.iterrows():
        topic = clean_text(row.get('pragraph', 'General'))
        question = clean_text(row.get('question', ''))
        answer = clean_text(row.get('answer', ''))
        
        if not question or not answer:
            logger.warning(f"Skipping row {idx}: Missing question or answer")
            continue
            
        # Generate stable ID
        doc_id = generate_stable_id(topic, question)
        
        # specific logic for topic-based ID prefix if desired (e.g. "DBS-001")
        # For now, using Topic-Hash
        
        entry = {
            "id": doc_id,
            "topic": topic,
            "question": question,
            "answer": answer,
            "metadata": {
                "tags": [topic], # Can expand this with keyword extraction later
                "source": input_file.name
            }
        }
        knowledge_base.append(entry)
        
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully converted {len(knowledge_base)} items to {output_path}")

if __name__ == "__main__":
    # Input: Root directory Q&A file
    # Output: app/resources/knowledge_base.json
    process_excel("Q&A_pairs.xlsx", "app/resources/knowledge_base.json")
