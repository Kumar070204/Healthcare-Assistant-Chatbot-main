import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import ollama
import logging
import time

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
                result = {
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][list(indices[0]).index(idx)])
                }
                results.append(result)
                logger.info(f"Retrieved chunk: {result['text'][:100]}... (Source: {result['metadata']['source']}, Distance: {result['distance']})")
            logger.info(f"Retrieved {len(results)} chunks")
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            return []

    def generate(self, query, retrieved, mode="qa", max_retries=3):
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
            
            for attempt in range(max_retries):
                try:
                    response = ollama.generate(model='gemma3:latest', prompt=prompt)['response']
                    logger.info("Response generated successfully")
                    return response
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error generating response after {max_retries} retries: {e}")
            return f"Error generating response: {str(e)}"

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
            return f"Error processing query: {str(e)}"

def interactive_query():
    rag = RAGPipeline()
    print("\nWelcome to the Health Chatbot!")
    print("Modes: qa (Q&A), fact_check (News Fact-Checking), myth (Myth-Busting)")
    print("Enter 'exit' to quit.\n")
    
    while True:
        mode = input("Select mode (qa, fact_check, myth): ").strip().lower()
        if mode == 'exit':
            print("Exiting...")
            break
        if mode not in ["qa", "fact_check", "myth"]:
            print("Invalid mode. Please choose: qa, fact_check, myth")
            continue
        
        query = input("Enter your query: ").strip()
        if query.lower() == 'exit':
            print("Exiting...")
            break
        if not query:
            print("Query cannot be empty.")
            continue
        
        print("\nProcessing...")
        response = rag.process_query(query, mode)
        print(f"\n{mode.upper()}: {response}\n")

if __name__ == "__main__":
    interactive_query()