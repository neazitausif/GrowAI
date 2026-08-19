

import requests
import json
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel



class SentimentAnalysis(BaseModel):
    
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float  # 0-1
    reasoning: str

class EntityExtractionResult(BaseModel):
    
    text: str
    entities: List[Dict[str, str]]  
    total_entities: int

class SummaryResult(BaseModel):
    
    original_text: str
    summary: str
    key_points: List[str]
    word_count_reduction: float  


class PromptEngineeringShowdown:
    def __init__(self, ollama_host="http://localhost:11434", model="qwen2:0.5b"):
        self.host = ollama_host
        self.model = model
        self.results = {}
        
        print(f"🚀 Prompt Engineering Showdown Initialized")
        print(f"   Model: {model}")
        print(f"   Host: {ollama_host}")
    
    def query_ollama(self, prompt: str, format_type=None) -> str:
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if format_type:
                payload["format"] = format_type
            
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Exception: {str(e)}"
    
    
    
    def sentiment_zero_shot(self, text: str) -> Dict:
        
        prompt = f"""Classify the sentiment of the following text as positive, negative, or neutral.

Text: {text}

Sentiment:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Zero-Shot",
            "prompt": prompt,
            "response": response,
            "category": "Sentiment Classification"
        }
    
    def sentiment_few_shot(self, text: str) -> Dict:
        
        prompt = f"""Classify the sentiment of text as positive, negative, or neutral.

Examples:
Text: "I love this product, it's absolutely amazing!"
Sentiment: positive

Text: "This service is terrible and waste of money."
Sentiment: negative

Text: "The weather today is cloudy."
Sentiment: neutral

Text: {text}

Sentiment:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Few-Shot (2-3 examples)",
            "prompt": prompt,
            "response": response,
            "category": "Sentiment Classification"
        }
    
    def sentiment_chain_of_thought(self, text: str) -> Dict:
        
        prompt = f"""Classify the sentiment of the text by reasoning step-by-step.

Text: {text}

Let's think step by step:
1. Identify emotional words in the text
2. Consider the overall tone
3. Determine if it's expressing satisfaction, dissatisfaction, or neutral tone
4. Make a final sentiment classification

Step-by-step reasoning:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Chain-of-Thought",
            "prompt": prompt,
            "response": response,
            "category": "Sentiment Classification"
        }
    
   
    
    def entity_extraction_zero_shot(self, text: str) -> Dict:
        
        prompt = f"""Extract named entities (person, location, organization, date) from the text.

Text: {text}

Entities:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Zero-Shot",
            "prompt": prompt,
            "response": response,
            "category": "Entity Extraction"
        }
    
    def entity_extraction_few_shot(self, text: str) -> Dict:
        
        prompt = f"""Extract named entities (person, location, organization, date) from text.

Examples:
Text: "Apple was founded by Steve Jobs in Cupertino, California in 1976."
Entities: Apple (organization), Steve Jobs (person), Cupertino (location), California (location), 1976 (date)

Text: "Einstein was born in Ulm, Germany on March 14, 1879."
Entities: Einstein (person), Ulm (location), Germany (location), March 14 1879 (date)

Text: {text}

Entities:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Few-Shot (2 examples)",
            "prompt": prompt,
            "response": response,
            "category": "Entity Extraction"
        }
    
    def entity_extraction_chain_of_thought(self, text: str) -> Dict:
        """Chain-of-thought entity extraction"""
        prompt = f"""Extract named entities by reasoning step-by-step.

Text: {text}

Let's think step by step:
1. Read through the text carefully
2. Identify all proper nouns (capitalized words)
3. Classify each as person, location, organization, date, or other
4. List all entities with their types

Step-by-step extraction:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Chain-of-Thought",
            "prompt": prompt,
            "response": response,
            "category": "Entity Extraction"
        }
    
    
    
    def summarization_zero_shot(self, text: str) -> Dict:
        """Zero-shot summarization"""
        prompt = f"""Summarize the following text in 1-2 sentences.

Text: {text}

Summary:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Zero-Shot",
            "prompt": prompt,
            "response": response,
            "category": "Text Summarization"
        }
    
    def summarization_few_shot(self, text: str) -> Dict:
        
        prompt = f"""Summarize text in 1-2 sentences capturing main ideas.

Examples:
Text: "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation. Guido van Rossum created Python in 1991 and it has become one of the most popular languages."
Summary: "Python is a popular, simple programming language created in 1991, widely used in web development, data science, AI, and automation."

Text: {text}

Summary:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Few-Shot (1 example)",
            "prompt": prompt,
            "response": response,
            "category": "Text Summarization"
        }
    
    def summarization_chain_of_thought(self, text: str) -> Dict:
        """Chain-of-thought summarization"""
        prompt = f"""Summarize the text by reasoning step-by-step.

Text: {text}

Let's think step by step:
1. Identify the main topic
2. Find the most important sentences
3. Extract key information
4. Combine into a concise summary

Step-by-step summary:"""
        
        response = self.query_ollama(prompt)
        return {
            "technique": "Chain-of-Thought",
            "prompt": prompt,
            "response": response,
            "category": "Text Summarization"
        }
    
   
    def demonstrate_failure_mode(self) -> Dict:
        
        print("\n" + "="*100)
        print(" DEMONSTRATING PROMPT FAILURE MODE")
        print("="*100)
        
       
        bad_prompt = "Analyze this text about machine learning. Tell me about it."
        
        print("\nBAD PROMPT (Vague, no structure):")
        print(f"  {bad_prompt}")
        
        response = self.query_ollama(bad_prompt)
        print("\nPROBLEM:")
        print("  - Output is unstructured")
        print("  - Inconsistent format")
        print("  - Can't parse results programmatically")
        
        return {
            "failure_mode": "Vague instruction with no format guidance",
            "bad_prompt": bad_prompt,
            "response": response
        }
    
    def fix_with_structured_output(self) -> Dict:
        
        print("\n" + "="*100)
        print(" FIXING WITH STRUCTURED OUTPUT")
        print("="*100)
        
        text = "Deep learning is amazing technology that powers AI systems."
        
      
        good_prompt = f"""Analyze this text and return ONLY valid JSON (no markdown, no extra text).

Text: "{text}"

Return JSON with this exact structure:
{{
  "text": "the input text",
  "main_topic": "extracted topic",
  "sentiment": "positive/negative/neutral",
  "key_concepts": ["concept1", "concept2"],
  "importance": 1-10
}}

JSON Response:"""
        
        print("\nIMPROVED PROMPT (Clear, structured):")
        print(f"  {good_prompt}")
        
        response = self.query_ollama(good_prompt)
        
        print("\nBENEFITS:")
        print("   Structured output")
        print("   Easy to parse")
        print("   Consistent format")
        print("   Can be used directly in applications")
        
        
        try:
            parsed = json.loads(response)
            print(f"\n Successfully parsed as JSON!")
            print(f"   Response: {json.dumps(parsed, indent=2)}")
        except:
            print(f"\n  Response couldn't be parsed as JSON (model may need refinement)")
            print(f"   Raw response: {response}")
        
        return {
            "fixed_prompt": good_prompt,
            "response": response
        }
    
    
    def run_full_showdown(self):
        
        print("\n" + "="*100)
        print(" RUNNING FULL PROMPT ENGINEERING SHOWDOWN")
        print("="*100)
        
       
        test_cases = {
            "sentiment": "The new iPhone is incredible, but the price is way too high!",
            "entities": "Steve Jobs founded Apple in Cupertino, California on April 1, 1976.",
            "summary": "Artificial intelligence has revolutionized technology. Machine learning enables computers to learn from data without explicit programming. Deep learning uses neural networks with multiple layers. These technologies power modern applications like recommendation systems, computer vision, and natural language processing."
        }
        
        
        print("\n TASK 1: SENTIMENT CLASSIFICATION")
        print("-" * 100)
        sentiment_results = [
            self.sentiment_zero_shot(test_cases["sentiment"]),
            self.sentiment_few_shot(test_cases["sentiment"]),
            self.sentiment_chain_of_thought(test_cases["sentiment"])
        ]
        self.display_comparison("Sentiment Classification", sentiment_results)
        
       
        print("\n TASK 2: ENTITY EXTRACTION")
        print("-" * 100)
        entity_results = [
            self.entity_extraction_zero_shot(test_cases["entities"]),
            self.entity_extraction_few_shot(test_cases["entities"]),
            self.entity_extraction_chain_of_thought(test_cases["entities"])
        ]
        self.display_comparison("Entity Extraction", entity_results)
        
        
        print("\n TASK 3: TEXT SUMMARIZATION")
        print("-" * 100)
        summary_results = [
            self.summarization_zero_shot(test_cases["summary"]),
            self.summarization_few_shot(test_cases["summary"]),
            self.summarization_chain_of_thought(test_cases["summary"])
        ]
        self.display_comparison("Text Summarization", summary_results)
        
       
        self.results = {
            "sentiment": sentiment_results,
            "entities": entity_results,
            "summary": summary_results
        }
    
    def display_comparison(self, task_name: str, results: List[Dict]):
        
        print(f"\nTask: {task_name}")
        print(f"Test Input: {results[0]['prompt'][:100]}...")
        print("-" * 100)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['technique']}")
            print(f"   Response: {result['response'][:150]}...")
        
        print("\n💡 OBSERVATIONS:")
        if results[0]['response'].strip() and results[1]['response'].strip():
            print("    Few-shot typically produces more structured output")
        if results[2]['response'].strip():
            print("    Chain-of-thought shows reasoning, often higher quality")
        print("    Combination of techniques gives best results")
    
    


def main():
    print("\n ASSIGNMENT 4: PROMPT ENGINEERING SHOWDOWN\n")
    
    showdown = PromptEngineeringShowdown(model="qwen2:0.5b")
    
    
    showdown.run_full_showdown()
    
    
    showdown.demonstrate_failure_mode()
    showdown.fix_with_structured_output()
    
    
    
    print("\n" + "="*100)
    print(" PROMPT ENGINEERING SHOWDOWN COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()
