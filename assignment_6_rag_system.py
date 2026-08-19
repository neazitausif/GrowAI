

import os
import numpy as np
from typing import List, Dict, Tuple
import json


try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
    import chromadb
    from chromadb.config import Settings
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install sentence-transformers rank-bm25 chromadb requests")
    DEPENDENCIES_OK = False


class DocumentChunker:
    
    
    
    def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks


class HybridSearchEngine:
   
    
    def __init__(self, model_name="all-MiniLM-L6-v2", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not DEPENDENCIES_OK:
            print(" Dependencies not available")
            return
        
        print(f" Initializing Hybrid Search Engine")
        print(f"   Embedding model: {model_name}")
        print(f"   Reranker model: {reranker_model}")
        
        self.embedding_model = SentenceTransformer(model_name)
        
        
        self.reranker = CrossEncoder(reranker_model)
        
        
        self.client = chromadb.Client()
        self.collection = None
        
        
        self.bm25 = None
        self.documents = []
        self.chunk_to_doc_id = {}
        
        print("Initialization complete!")
    
    def load_and_chunk_document(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        
        print("\n DOCUMENT INGESTION")
        print("-" * 100)
        
        chunks = DocumentChunker.chunk_text(text, chunk_size, overlap)
        
        print(f"Document loaded")
        print(f"  Total length: {len(text)} characters")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Overlap: {overlap}")
        print(f"  Number of chunks: {len(chunks)}")
        
        return chunks
    
    def build_indices(self, chunks: List[str]):
        
        print("\n BUILDING INDICES")
        print("-" * 100)
        
        self.documents = chunks
        
        
        print("Creating vector embeddings...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[{"chunk_id": i}]
            )
        
        print(f" Created ChromaDB index with {len(chunks)} chunks")
        
        
        print("Creating BM25 index...")
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        print(f"Created BM25 index with {len(chunks)} chunks")
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
       
        return [
            (int(results['ids'][0][i]), 1 - results['distances'][0][i], results['documents'][0][i])
            for i in range(len(results['ids'][0]))
        ]
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        
        return [
            (idx, scores[idx], self.documents[idx])
            for idx in top_indices if scores[idx] > 0
        ]
    
    def reciprocal_rank_fusion(self, vector_results: List[Tuple[int, float, str]], 
                               bm25_results: List[Tuple[int, float, str]], 
                               k: int = 60) -> List[Tuple[int, float, str]]:
        
        
        
        rrf_scores = {}
        
        
        for rank, (chunk_id, score, text) in enumerate(vector_results):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"score": 0, "text": text}
            rrf_scores[chunk_id]["score"] += 1 / (k + rank + 1)
        
        
        for rank, (chunk_id, score, text) in enumerate(bm25_results):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"score": 0, "text": text}
            rrf_scores[chunk_id]["score"] += 1 / (k + rank + 1)
        
        
        sorted_results = sorted(
            [(chunk_id, data["score"], data["text"]) for chunk_id, data in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def rerank_results(self, query: str, results: List[Tuple[int, float, str]], 
                      top_k: int = 3) -> List[Tuple[int, float, str]]:
        
        print(f"\n RERANKING {len(results)} results with cross-encoder...")
        
       
        pairs = [[query, result[2]] for result in results]
        
        
        scores = self.reranker.predict(pairs)
        
       
        scored_results = [
            (results[i][0], scores[i], results[i][2])
            for i in range(len(results))
        ]
        
        
        sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[str]:
        
        print(f"\n HYBRID SEARCH FOR: '{query}'")
        print("-" * 100)
        
        
        print("1. Vector search...")
        vector_results = self.vector_search(query, top_k)
        print(f"   Found {len(vector_results)} results")
        
       
        print("2. BM25 keyword search...")
        bm25_results = self.bm25_search(query, top_k)
        print(f"   Found {len(bm25_results)} results")
        
        
        print("3. Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        print(f"   Fused into {len(fused_results)} unique results")
        
       
        print("4. Cross-encoder reranking...")
        reranked = self.rerank_results(query, fused_results, top_k=3)
        print(f"   Selected top 3 results")
        
        print("\n FINAL RESULTS:")
        print("-" * 100)
        for rank, (chunk_id, score, text) in enumerate(reranked, 1):
            print(f"\n{rank}. Score: {score:.4f} | Chunk {chunk_id}")
            print(f"   {text[:100]}...")
        
        return [result[2] for result in reranked]
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        
        print("\n GENERATING ANSWER WITH LLM")
        print("-" * 100)
        
       
        context = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        
        prompt = f"""Answer the question using ONLY the context provided. 
If the answer is not found in the context, say 'Not in context'.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        print(f"Context length: {len(context)} characters")
        print("Querying LLM...")
        
        
        answer = """Based on the provided context, the answer is: [LLM would generate answer here]
Note: In production, this would call an LLM API (Ollama, OpenAI, etc.)"""
        
        return answer


def create_sample_document() -> str:
   
    
    document = """
    Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods 
    from carbon dioxide and water. This process is vital to life on Earth since it produces oxygen and converts 
    solar energy into chemical energy that can be stored in glucose molecules.
    
    The Light-Dependent Reactions:
    These reactions occur in the thylakoid membranes of the chloroplasts. During these reactions, light energy 
    is captured by chlorophyll molecules and converted into chemical energy in the form of ATP and NADPH. 
    Water molecules are split, releasing oxygen as a byproduct. The light-dependent reactions can be summarized 
    as: 2H2O + 2NADP+ + 3ADP + 3Pi → O2 + 2NADPH + 3ATP
    
    The Light-Independent Reactions (Calvin Cycle):
    Also known as the dark reactions or light-independent reactions, these take place in the stroma of the chloroplast. 
    The Calvin cycle uses the ATP and NADPH produced in the light-dependent reactions to fix carbon dioxide into 
    three-carbon sugars. The cycle has three phases: carbon fixation, reduction, and regeneration of RuBP.
    
    Factors Affecting Photosynthesis:
    1. Light Intensity: Increasing light intensity increases the rate of photosynthesis until saturation point.
    2. Carbon Dioxide Concentration: Higher CO2 concentrations increase photosynthesis rates.
    3. Temperature: Photosynthesis is most efficient at temperatures between 25-35°C.
    4. Chlorophyll Content: More chlorophyll allows more light absorption.
    5. Water Availability: Water is essential as it's a raw material for the light-dependent reactions.
    
    Types of Photosynthesis:
    C3 Photosynthesis: The most common form, found in most plants. The first stable compound has 3 carbons.
    C4 Photosynthesis: Found in plants like corn and sugarcane. More efficient in hot, dry climates.
    CAM Photosynthesis: Found in succulents and desert plants. Stomata open at night to conserve water.
    
    Importance of Photosynthesis:
    - Produces atmospheric oxygen necessary for most life forms
    - Converts solar energy into chemical energy stored in organic compounds
    - Forms the base of most food chains and food webs
    - Helps regulate atmospheric carbon dioxide levels
    - Critical for human survival and agriculture
    """
    
    return document


def main():
    print("\n ASSIGNMENT 6: DOCUMENT Q&A SYSTEM WITH HYBRID SEARCH\n")
    
    if not DEPENDENCIES_OK:
        print("  Dependencies not available. Please install:")
        print("   pip install sentence-transformers rank-bm25 chromadb")
        return
    
   
    print(" Loading sample document...")
    document = create_sample_document()
    print(f" Loaded {len(document)} character document\n")
    
   
    engine = HybridSearchEngine()
    
    
    chunks = engine.load_and_chunk_document(document, chunk_size=300, overlap=50)
    
    engine.build_indices(chunks)
    
    
    test_queries = [
        "What is photosynthesis?",  # Semantic question
        "light reactions ATP",      # Keyword-heavy query
        "temperature CO2 photosynthesis", # Multiple keywords
        "what are the types of photosynthesis?",  # Specific information
        "how does CAM help desert plants?",  # Reasoning question
    ]
    
    print("\n" + "="*100)
    print(" TESTING HYBRID SEARCH WITH 5 QUERIES")
    print("="*100)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*100}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*100}")
        
        
        context_chunks = engine.hybrid_search(query)
        
        
        answer = engine.generate_answer(query, context_chunks)
        print(f"\n{answer}\n")
    
    
    print("\n" + "="*100)
   


if __name__ == "__main__":
    main()
