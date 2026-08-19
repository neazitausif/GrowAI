

import tiktoken
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

class TokenExplorer:
    def __init__(self):
        
        print("Initializing tokenizers...")
        self.tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        self.hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        
        
        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        
        self.reference_sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
            "Python is a popular programming language.",
            "Deep learning uses neural networks with multiple layers.",
            "Coffee is the most consumed beverage worldwide."
        ]
        
        
        self.reference_embeddings = self.embedding_model.encode(
            self.reference_sentences, 
            convert_to_numpy=True
        )
    
    def tokenize_text(self, text):
        
        print("\n" + "="*80)
        print(f"INPUT TEXT: {text}")
        print("="*80)
        
       
        tiktoken_ids = self.tiktoken_enc.encode(text)
        tiktoken_tokens = [self.tiktoken_enc.decode_single_token_bytes(id).decode('utf-8', errors='ignore') 
                           for id in tiktoken_ids]
        
        
        hf_encoded = self.hf_tokenizer.encode(text)
        hf_tokens = [self.hf_tokenizer.decode([id]) for id in hf_encoded]
        
        
        print(f"\n TOKENIZATION COMPARISON:")
        print(f"\nTikToken (cl100k_base):")
        print(f"  Total Tokens: {len(tiktoken_ids)}")
        print(f"  Tokens: {tiktoken_tokens}")
        print(f"  Token IDs: {tiktoken_ids}")
        
        print(f"\nHuggingFace (Qwen2-0.5B):")
        print(f"  Total Tokens: {len(hf_encoded)}")
        print(f"  Tokens: {hf_tokens}")
        print(f"  Token IDs: {hf_encoded}")
        
        print(f"\n ANALYSIS:")
        print(f"  TikToken uses {len(tiktoken_ids)} tokens")
        print(f"  HuggingFace uses {len(hf_encoded)} tokens")
        print(f"  Difference: {abs(len(tiktoken_ids) - len(hf_encoded))} tokens")
        
        return text
    
    def compute_semantic_similarity(self, text):
       
        print("\n" + "="*80)
        print(" SEMANTIC SIMILARITY ANALYSIS")
        print("="*80)
        
        
        input_embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        
        similarities = cosine_similarity([input_embedding], self.reference_embeddings)[0]
        
       
        ranked = sorted(
            zip(self.reference_sentences, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\nInput: '{text}'")
        print(f"\nRanked Similarity Scores:")
        print("-" * 80)
        for rank, (sentence, score) in enumerate(ranked, 1):
            print(f"{rank}. Score: {score:.4f} | '{sentence}'")
        
        print("\n INTERPRETATION:")
        print(f"  Highest similarity: {ranked[0][1]:.4f} - '{ranked[0][0]}'")
        print(f"  Lowest similarity: {ranked[-1][1]:.4f} - '{ranked[-1][0]}'")
        
        return ranked
    
    def process_input(self, text):
        
        self.tokenize_text(text)
        similarities = self.compute_semantic_similarity(text)
        return similarities


def main():
    explorer = TokenExplorer()
    
    
    print("\n" + " TEST CASE 1: English Technology Text" + "\n")
    test1 = "Artificial intelligence and machine learning are transforming technology."
    explorer.process_input(test1)
    
    
    print("\n" + " TEST CASE 2: English General Text" + "\n")
    test2 = "The weather affects our daily activities."
    explorer.process_input(test2)
    
    
    print("\n" + " TEST CASE 3: Chinese (Non-English)" + "\n")
    test3 = "人工智能是未来的技术。"  # "Artificial intelligence is the technology of the future"
    explorer.process_input(test3)
    
    print("\n" + "="*80)
    print(" TOKEN EXPLORER ANALYSIS COMPLETE")
    print("="*80)
    
    


if __name__ == "__main__":
    main()
