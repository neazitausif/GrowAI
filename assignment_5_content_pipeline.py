
import os
import sys


try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print(" LangChain not installed. Install with:")
    print("   pip install langchain langchain-ollama")
    LANGCHAIN_AVAILABLE = False


try:
    from llama_index.llms.ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    print("LlamaIndex not installed. Install with:")
    print("   pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False


class LangChainPipeline:
    
    
    def __init__(self, model_name="qwen2:0.5b", ollama_host="http://localhost:11434"):
        if not LANGCHAIN_AVAILABLE:
            print("LangChain not available")
            return
        
        print(f"Initializing LangChain Pipeline")
        print(f"   Model: {model_name}")
        print(f"   Host: {ollama_host}")
        
        
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_host,
            temperature=0.7
        )
        
        self.model_name = model_name
        self.output_parser = StrOutputParser()
    
    def build_pipeline(self):
        
        print("\n BUILDING LANGCHAIN PIPELINE")
        print("-" * 100)
        
        
        print("Step 1: Creating explanation generator...")
        explanation_template = ChatPromptTemplate.from_messages([
            ("system", "You are a teacher explaining complex topics to a 10-year-old child. Use simple words and fun examples."),
            ("user", "Explain {topic} in 200 words that a 10-year-old would understand.")
        ])
        
        explanation_chain = explanation_template | self.llm | self.output_parser
        
        
        print("Step 2: Creating quiz generator...")
        quiz_template = ChatPromptTemplate.from_messages([
            ("system", "You are a teacher creating quiz questions. Make questions clear and appropriate for the topic."),
            ("user", "Based on this explanation, create 5 quiz questions:\n\n{explanation}\n\nProvide just the numbered questions, one per line.")
        ])
        
        quiz_chain = quiz_template | self.llm | self.output_parser
        
        
        print("Step 3: Creating answer key generator...")
        answer_template = ChatPromptTemplate.from_messages([
            ("system", "You are a teacher creating answer keys. Provide clear, accurate answers."),
            ("user", "Create an answer key for these quiz questions:\n\n{quiz}\n\nProvide concise answers.")
        ])
        
        answer_chain = answer_template | self.llm | self.output_parser
        
        return explanation_chain, quiz_chain, answer_chain
    
    def run(self, topic="photosynthesis"):
        
        if not LANGCHAIN_AVAILABLE:
            print("Cannot run - LangChain not available")
            return None
        
        print("\n" + "="*100)
        print(" RUNNING LANGCHAIN PIPELINE")
        print("="*100)
        print(f"Topic: {topic}\n")
        
        explanation_chain, quiz_chain, answer_chain = self.build_pipeline()
        
        try:
           
            print("Step 1: Generating explanation...")
            explanation = explanation_chain.invoke({"topic": topic})
            print("Explanation generated!\n")
            print("EXPLANATION:")
            print("-" * 100)
            print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
            
            
            print("\n Step 2: Generating quiz questions...")
            quiz = quiz_chain.invoke({"explanation": explanation})
            print(" Quiz generated!\n")
            print(" QUIZ QUESTIONS:")
            print("-" * 100)
            print(quiz)
            
            
            print("\n Step 3: Generating answer key...")
            answers = answer_chain.invoke({"quiz": quiz})
            print(" Answer key generated!\n")
            print(" ANSWER KEY:")
            print("-" * 100)
            print(answers)
            
            return {
                "topic": topic,
                "explanation": explanation,
                "quiz": quiz,
                "answers": answers
            }
        
        except Exception as e:
            print(f" Error running pipeline: {e}")
            return None


class LlamaIndexPipeline:
    
    
    def __init__(self, model_name="qwen2:0.5b", ollama_host="http://localhost:11434"):
        if not LLAMAINDEX_AVAILABLE:
            print(" LlamaIndex not available")
            return
        
        print(f" Initializing LlamaIndex Pipeline")
        print(f"   Model: {model_name}")
        print(f"   Host: {ollama_host}")
        
        
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_host,
            temperature=0.7
        )
        
        self.model_name = model_name
    
    def run(self, topic="photosynthesis"):
        
        if not LLAMAINDEX_AVAILABLE:
            print("Cannot run - LlamaIndex not available")
            return None
        
        print("\n" + "="*100)
        print("RUNNING LLAMAINDEX PIPELINE")
        print("="*100)
        print(f"Topic: {topic}\n")
        
        try:
           
            print("Step 1: Generating explanation...")
            explanation_prompt = f"""You are a teacher explaining complex topics to a 10-year-old child. 
Use simple words and fun examples.

Explain {topic} in 200 words that a 10-year-old would understand."""
            
            explanation = self.llm.complete(explanation_prompt).text
            print("Explanation generated!\n")
            print("EXPLANATION:")
            print("-" * 100)
            print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
            
            
            print("\n Step 2: Generating quiz questions...")
            quiz_prompt = f"""You are a teacher creating quiz questions. Make questions clear and appropriate.

Based on this explanation, create 5 quiz questions:

{explanation}

Provide just the numbered questions, one per line."""
            
            quiz = self.llm.complete(quiz_prompt).text
            print("Quiz generated!\n")
            print("QUIZ QUESTIONS:")
            print("-" * 100)
            print(quiz)
            
            
            print("\n⏳ Step 3: Generating answer key...")
            answer_prompt = f"""You are a teacher creating answer keys. Provide clear, accurate answers.

Create an answer key for these quiz questions:

{quiz}

Provide concise answers."""
            
            answers = self.llm.complete(answer_prompt).text
            print("Answer key generated!\n")
            print("ANSWER KEY:")
            print("-" * 100)
            print(answers)
            
            return {
                "topic": topic,
                "explanation": explanation,
                "quiz": quiz,
                "answers": answers
            }
        
        except Exception as e:
            print(f"Error running pipeline: {e}")
            return None



   
    
   



    


def main():
    print("\n ASSIGNMENT 5: MULTI-STEP CONTENT PIPELINE\n")
    print("This assignment compares LangChain and LlamaIndex\n")
    
    topic = "photosynthesis"
    
    
    if LANGCHAIN_AVAILABLE:
        print("\n" + "="*100)
        print("1 LANGCHAIN PIPELINE")
        print("="*100)
        langchain_pipeline = LangChainPipeline()
        langchain_results = langchain_pipeline.run(topic)
    else:
        print("\n LangChain not available - skipping LangChain demonstration")
        langchain_results = None
    
    
    if LLAMAINDEX_AVAILABLE:
        print("\n" + "="*100)
        print(" LLAMAINDEX PIPELINE")
        print("="*100)
        llamaindex_pipeline = LlamaIndexPipeline()
        llamaindex_results = llamaindex_pipeline.run(topic)
    else:
        print("\n LlamaIndex not available - skipping LlamaIndex demonstration")
        llamaindex_results = None
    
   
    
    print("\n" + "="*100)
    print("MULTI-STEP CONTENT PIPELINE COMPLETE!")
    print("="*100)
    


if __name__ == "__main__":
    main()
