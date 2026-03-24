import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import ollama
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAGPipeline...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index("data/db/faiss_index.index")
            with open("data/db/metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            with open("data/db/chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            logger.info("RAGPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAGPipeline: {e}")
            raise

    def retrieve(self, query, top_k=5, filter_type=None):
        logger.info(f"Retrieving for query: {query}")
        try:
            query_emb = self.model.encode([query], convert_to_tensor=False).astype('float32')
            distances, indices = self.index.search(query_emb, top_k)
            results = []
            for idx in indices[0]:
                if filter_type and self.metadata[idx]['type'] != filter_type:
                    continue
                results.append({
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][list(indices[0]).index(idx)])
                })
            logger.info(f"Retrieved {len(results)} chunks")
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            return []

    def generate(self, query, retrieved, mode="qa"):
        logger.info(f"Generating response for query: {query} (mode: {mode})")
        try:
            context = "\n".join([r["text"] for r in retrieved])
            prompt = f"Query: {query}\nContext: {context}\n"
            if mode == "qa":
                prompt += "Provide an accurate, concise answer in MedQuAD style."
            elif mode == "fact_check":
                prompt += "Cross-check the query against reliable sources. If false, debunk with sources."
            elif mode == "myth":
                prompt += "Check if the query is a myth. If false, debunk using MB-KB. Cite sources."
            
            response = ollama.generate(model='gemma3:latest', prompt=prompt)['response']
            logger.info("Response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"

    def process_query(self, query, mode="qa"):
        logger.info(f"Processing query: {query} (mode: {mode})")
        try:
            if mode == "qa":
                intent_chunks = self.retrieve(query, filter_type="query_intent")
                if intent_chunks:
                    logger.info("Query intent matched; proceeding with original query")
            retrieved = self.retrieve(query, filter_type="myth" if mode in ["myth", "fact_check"] else None)
            return self.generate(query, retrieved, mode)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Error processing query"

if __name__ == "__main__":
    rag = RAGPipeline()
    test_queries = [
        ("What are the causes of asthma?", "qa"),
        ("Garlic cures COVID", "fact_check"),
        ("Drinking hot water prevents COVID", "myth")
    ]
    for query, mode in test_queries:
        logger.info(f"Testing query: {query} (mode: {mode})")
        print(f"{mode.upper()}: {rag.process_query(query, mode)}")