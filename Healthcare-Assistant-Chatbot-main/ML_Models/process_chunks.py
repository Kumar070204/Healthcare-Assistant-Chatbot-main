import pandas as pd
import json
import os
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")

def process_medquad():
    logger.info("Processing MedQuAD...")
    chunks = []
    try:
        df = pd.read_csv("data/raw/medquad.csv")
        for _, row in df.iterrows():
            chunk = f"Q: {row['question']}\nA: {row['answer']}"
            chunks.append({"text": chunk, "metadata": {"source": "MedQuAD", "type": "qa"}})
        logger.info(f"Processed {len(chunks)} MedQuAD chunks")
    except Exception as e:
        logger.error(f"Error processing MedQuAD: {e}")
    return chunks

def process_healthcaremagic():
    logger.info("Processing HealthcareMagic...")
    chunks = []
    try:
        df = pd.read_csv("data/raw/healthcaremagic_questions.csv")
        for _, row in df.iterrows():
            chunks.append({"text": str(row['Question']), "metadata": {"source": "HealthcareMagic", "type": "query_intent"}})
        logger.info(f"Processed {len(chunks)} HealthcareMagic chunks")
    except Exception as e:
        logger.error(f"Error processing HealthcareMagic: {e}")
    return chunks

def process_mb_kb():
    logger.info("Processing MB-KB...")
    chunks = []
    try:
        with open("data/processed/mb_kb.json", "r") as f:
            mb_data = json.load(f)
        for item in mb_data:
            chunk = f"Claim: {item['claim']}\nStatus: {item['status']}\nInfo: {item['info']}"
            chunks.append({"text": chunk, "metadata": {"source": item['source'], "type": "myth"}})
        logger.info(f"Processed {len(chunks)} MB-KB chunks")
    except Exception as e:
        logger.error(f"Error processing MB-KB: {e}")
    return chunks

def save_chunks():
    logger.info("Saving chunks...")
    try:
        ensure_dir("data/processed")
        chunks = process_medquad() + process_healthcaremagic() + process_mb_kb()
        with open("data/processed/all_chunks.json", "w") as f:
            json.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks to data/processed/all_chunks.json")
        return chunks
    except Exception as e:
        logger.error(f"Error saving chunks: {e}")
        return []

def embed_and_index():
    logger.info("Embedding and indexing chunks...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with open("data/processed/all_chunks.json", "r") as f:
            chunks = json.load(f)
        
        texts = [c['text'] for c in chunks]
        logger.info("Encoding texts...")
        embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        logger.info("Adding embeddings to FAISS index...")
        index.add(embeddings)

        ensure_dir("data/db")
        faiss.write_index(index, "data/db/faiss_index.index")
        with open("data/db/metadata.pkl", "wb") as f:
            pickle.dump([c['metadata'] for c in chunks], f)
        with open("data/db/chunks.pkl", "wb") as f:
            pickle.dump([c['text'] for c in chunks], f)
        logger.info("FAISS index and metadata saved successfully")
    except Exception as e:
        logger.error(f"Error embedding/indexing: {e}")

if __name__ == "__main__":
    save_chunks()
    embed_and_index()