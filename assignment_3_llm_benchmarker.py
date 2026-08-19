

import requests
import json
import csv
from datetime import datetime
import time
from typing import List, Dict

class OllamaLLMBenchmarker:
    def __init__(self, ollama_host="http://localhost:11434"):
        self.host = ollama_host
        self.models = []
        self.results = []
        
        
        self.prompts = {
            "factual": {
                "text": "What is the capital of France?",
                "category": "Simple Factual Question"
            },
            "sentiment": {
                "text": "Classify the sentiment of this text: 'I love this product, it's absolutely amazing!' as positive, negative, or neutral.",
                "category": "Sentiment Classification"
            },
            "reasoning": {
                "text": "If there are 3 apples and you eat 1, then buy 2 more, how many apples do you have? Explain your reasoning.",
                "category": "Reasoning Problem"
            },
            "code": {
                "text": "Write a Python function that returns the nth Fibonacci number.",
                "category": "Code Generation"
            },
            "creative": {
                "text": "Write a short creative story (2-3 sentences) about a time traveler discovering the internet in 1995.",
                "category": "Creative Writing"
            }
        }
        
        
        self.model_configs = [
            {"name": "tinyllama", "size": "Small (1.1B)"},
            {"name": "qwen2:0.5b", "size": "Tiny (0.5B)"},
            {"name": "phi3:mini", "size": "Small (3.8B)"},
        ]
        
        print(" LLM BENCHMARKER INITIALIZED")
        print(f"   Host: {self.host}")
        print(f"   Prompts: {len(self.prompts)} categories")
        print(f"   Models to test: {[m['name'] for m in self.model_configs]}")
    
    def verify_ollama_connection(self):
        
        print("\n Verifying Ollama connection...")
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                print(f"    Ollama is running!")
                print(f"   Available models: {len(available_models)}")
                for model in available_models:
                    print(f"      - {model.get('name')}")
                return True
            else:
                print("    Ollama connection failed")
                return False
        except Exception as e:
            print(f"    Error connecting to Ollama: {e}")
            print("   Please ensure Ollama is running: ollama serve")
            return False
    
    def pull_model(self, model_name: str) -> bool:
        """Pull model if not already present"""
        print(f"\n Ensuring {model_name} is available...")
        try:
            
            tags_response = requests.get(f"{self.host}/api/tags")
            existing_models = [m['name'] for m in tags_response.json().get('models', [])]
            
            if any(model_name in m for m in existing_models):
                print(f"    {model_name} already available")
                return True
            
            print(f"     Pulling {model_name}... (this may take a few minutes)")
            response = requests.post(f"{self.host}/api/pull", json={"name": model_name})
            
            if response.status_code == 200:
                print(f"    Successfully pulled {model_name}")
                return True
            else:
                print(f"     Failed to pull {model_name}")
                return False
        except Exception as e:
            print(f"     Error pulling model: {e}")
            return False
    
    def query_model(self, model_name: str, prompt: str) -> Dict:
        
        try:
            url = f"{self.host}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=300)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                elapsed_ms = (end_time - start_time) * 1000
                
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "duration_ms": elapsed_ms,
                    "total_duration": data.get("total_duration", 0) / 1_000_000,  # Convert to ms
                    "tokens_per_sec": data.get("eval_count", 0) / (data.get("total_duration", 1) / 1_000_000_000)
                }
            else:
                return {
                    "success": False,
                    "response": f"Error: {response.status_code}",
                    "duration_ms": (end_time - start_time) * 1000
                }
        except Exception as e:
            return {
                "success": False,
                "response": f"Exception: {str(e)}",
                "duration_ms": 0
            }
    
    def rate_response_quality(self, response: str, category: str) -> int:
        """
        Manually rate response quality on scale 1-5
        In practice, this would use an automated evaluator
        """
       
        score = 3  
        
       
        if len(response) < 20:
            score = 1
        elif len(response) < 50:
            score = 2
        elif len(response) < 200:
            score = 3
        elif len(response) < 500:
            score = 4
        else:
            score = 5
        
       
        if category == "factual" and ("France" in response or "Paris" in response):
            score = 5
        elif category == "sentiment" and ("positive" in response.lower()):
            score = 4
        elif category == "reasoning" and ("3" in response or "three" in response.lower()):
            score = 4
        elif category == "code" and ("def " in response or "Fibonacci" in response):
            score = 4
        elif category == "creative" and len(response) > 100:
            score = 4
        
        return score
    
    def run_benchmark(self):
        
        if not self.verify_ollama_connection():
            print("\n Skipping benchmark - Ollama not available")
            return
        
        print("\n" + "="*100)
        print(" STARTING BENCHMARK")
        print("="*100)
        
        total_tests = len(self.model_configs) * len(self.prompts)
        current_test = 0
        
        for model_config in self.model_configs:
            model_name = model_config["name"]
            model_size = model_config["size"]
            
            print(f"\n Testing Model: {model_name} ({model_size})")
            print("-" * 100)
            
            
            if not self.pull_model(model_name):
                print(f"    Skipping {model_name} - could not pull")
                continue
            
            for prompt_key, prompt_data in self.prompts.items():
                current_test += 1
                prompt_text = prompt_data["text"]
                category = prompt_data["category"]
                
                print(f"   [{current_test}/{total_tests}] {category}...", end=" ", flush=True)
                
                
                result = self.query_model(model_name, prompt_text)
                
                if result["success"]:
                    
                    quality_score = self.rate_response_quality(result["response"], prompt_key)
                    
                    
                    response_preview = result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"]
                    
                    print(f" ({result['duration_ms']:.1f}ms, Quality: {quality_score}/5)")
                    
                    
                    self.results.append({
                        "model": model_name,
                        "model_size": model_size,
                        "prompt_category": category,
                        "prompt_text": prompt_text,
                        "response": result["response"],
                        "response_time_ms": round(result["duration_ms"], 2),
                        "quality_score": quality_score,
                        "tokens_per_sec": round(result.get("tokens_per_sec", 0), 2),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    print(f" (Failed)")
                    self.results.append({
                        "model": model_name,
                        "model_size": model_size,
                        "prompt_category": category,
                        "response": result["response"],
                        "response_time_ms": round(result["duration_ms"], 2),
                        "quality_score": 0,
                        "timestamp": datetime.now().isoformat()
                    })
    
    def generate_report(self):
        
        if not self.results:
            print("\n  No results to report")
            return
        
        print("\n" + "="*100)
        print(" BENCHMARK REPORT")
        print("="*100)
        
        
        print("\n SUMMARY BY MODEL:")
        print("-" * 100)
        print(f"{'Model':<20} {'Size':<15} {'Avg Time (ms)':<15} {'Avg Quality':<15} {'Throughput':<15}")
        print("-" * 100)
        
        models_summary = {}
        for result in self.results:
            model = result["model"]
            if model not in models_summary:
                models_summary[model] = {
                    "times": [],
                    "qualities": [],
                    "size": result["model_size"]
                }
            if "response_time_ms" in result and result["response_time_ms"] > 0:
                models_summary[model]["times"].append(result["response_time_ms"])
            if "quality_score" in result:
                models_summary[model]["qualities"].append(result["quality_score"])
        
        for model, data in models_summary.items():
            avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
            avg_quality = sum(data["qualities"]) / len(data["qualities"]) if data["qualities"] else 0
            throughput = (1000 / avg_time) if avg_time > 0 else 0  # Responses per second
            
            print(f"{model:<20} {data['size']:<15} {avg_time:<15.2f} {avg_quality:<15.2f} {throughput:<15.2f}")
        
        
        print("\n SUMMARY BY PROMPT CATEGORY:")
        print("-" * 100)
        print(f"{'Category':<25} {'Avg Time (ms)':<15} {'Avg Quality':<15} {'Best Model':<20}")
        print("-" * 100)
        
        categories_summary = {}
        for result in self.results:
            cat = result["prompt_category"]
            if cat not in categories_summary:
                categories_summary[cat] = {"times": [], "qualities": [], "models": []}
            if result["response_time_ms"] > 0:
                categories_summary[cat]["times"].append(result["response_time_ms"])
                categories_summary[cat]["models"].append(result["model"])
            categories_summary[cat]["qualities"].append(result["quality_score"])
        
        for category, data in categories_summary.items():
            avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
            avg_quality = sum(data["qualities"]) / len(data["qualities"]) if data["qualities"] else 0
            best_model = data["models"][0] if data["models"] else "N/A"
            
            print(f"{category:<25} {avg_time:<15.2f} {avg_quality:<15.2f} {best_model:<20}")
    
    def save_to_csv(self, filename="benchmark_results.csv"):
       
        if not self.results:
            print("  No results to save")
            return
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            
            print(f"\n Results saved to {filename}")
        except Exception as e:
            print(f"\n Error saving CSV: {e}")
    
    def make_recommendations(self):
        
        print("\n" + "="*100)
        
        print("="*100)
        
        if not self.results:
            return
        
        models_summary = {}
        for result in self.results:
            model = result["model"]
            if model not in models_summary:
                models_summary[model] = {"times": [], "qualities": []}
            if "response_time_ms" in result and result["response_time_ms"] > 0:
                models_summary[model]["times"].append(result["response_time_ms"])
            if "quality_score" in result:
                models_summary[model]["qualities"].append(result["quality_score"])
        
        recommendations = """
        MODEL SELECTION FRAMEWORK:
        
        1. SPEED IS PRIORITY (Real-time applications):
           → Use: Smallest/fastest model (TinyLLaMA, Qwen2:0.5B)
           → Trade-off: Lower quality responses
           → Use case: Chatbots, real-time suggestions, low-latency APIs
        
        2. QUALITY IS PRIORITY (Accuracy-critical tasks):
           → Use: Larger models (Phi3, better instruction-tuned models)
           → Trade-off: Slower responses, more compute needed
           → Use case: Report generation, complex analysis, customer-facing
        
        3. BALANCED APPROACH (Most production scenarios):
           → Use: Medium-sized models with good quality/speed ratio
           → Strategy: Multi-tier system - use fast model for simple queries,
                      fall back to larger model for complex reasoning
           → Use case: Customer support, general-purpose assistants
        
        COST CONSIDERATIONS:
           • Smaller models: Lower latency, less GPU VRAM, cheaper inference
           • Larger models: Higher quality, more reliable, but slower
           • On-premise (Ollama): No API costs, full privacy, requires hardware
        
        SCALING STRATEGIES:
           • Start small: Test on Qwen2:0.5B first
           • Measure latency: Understanding response time is critical
           • User testing: Quality ratings matter more than raw metrics
           • A/B testing: Compare models in production with real users
        
        NEXT STEPS:
           • Monitor quality/speed tradeoff with your actual workload
           • Consider model fine-tuning for specific domains
           • Implement caching for frequent queries
           • Use larger models for batch processing, smaller for interactive
        """
        
        print(recommendations)


def main():
    print("\n ASSIGNMENT 3: LOCAL LLM BENCHMARKER\n")
    
    
    benchmarker = OllamaLLMBenchmarker()
    
    
    benchmarker.run_benchmark()
    
    
    benchmarker.generate_report()
    
    
    benchmarker.save_to_csv("/home/claude/benchmark_results.csv")
    
    benchmarker.make_recommendations()
    
    print("\n" + "="*100)
    print(" BENCHMARK COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()
