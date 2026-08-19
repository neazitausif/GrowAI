

import json
from datetime import datetime
from typing import Any, Dict, List
from dataclasses import dataclass

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("  LangGraph not installed. Install with:")
    print("   pip install langgraph langchain langchain-ollama")
    LANGGRAPH_AVAILABLE = False


@dataclass
class ToolCall:
    
    tool_name: str
    tool_input: str
    tool_output: str
    reasoning: str



@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Math expression like '2 + 2' or 'sqrt(16)'
    
    Returns:
        Result of the calculation
    """
    try:
        
        safe_dict = {
            'sqrt': lambda x: x ** 0.5,
            '__builtins__': {}
        }
        result = eval(expression, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def define_word(word: str) -> str:
    """
    Get definition of a word.
    
    Args:
        word: The word to define
    
    Returns:
        Definition of the word
    """
    
    definitions = {
        "photosynthesis": "Process by which plants convert sunlight to chemical energy",
        "artificial": "Made by humans, not natural",
        "intelligence": "Ability to learn, understand, and apply knowledge",
        "algorithm": "Step-by-step procedure for solving a problem",
        "data": "Information or facts, typically in numerical form",
        "quantum": "A discrete quantity or amount",
        "machine": "A mechanical device that performs tasks",
        "learning": "Process of acquiring knowledge or skills",
        "neural": "Related to the nervous system or artificial neurons",
        "network": "Connected system of computers or nodes"
    }
    
    word_lower = word.lower()
    if word_lower in definitions:
        return f"'{word}' means: {definitions[word_lower]}"
    else:
        return f"Definition not found for '{word}'. Try a different word."



def get_current_datetime() -> str:
    
    now = datetime.now()
    return now.strftime("Current date and time: %A, %B %d, %Y at %I:%M %p")



class ReActAgent:
    
    
    def __init__(self, model_name: str = "qwen2:0.5b", ollama_host: str = "http://localhost:11434"):
        if not LANGGRAPH_AVAILABLE:
            print(" LangGraph not available")
            return
        
        print(f"   Initializing ReAct Agent")
        print(f"   Model: {model_name}")
        print(f"   Host: {ollama_host}")
        
       
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_host,
            temperature=0.7
        )
        
       
        self.tools = [calculate, define_word, get_current_datetime]
        
        
        memory = MemorySaver()
        
        try:
            self.agent_executor = create_react_agent(
                self.llm,
                self.tools,
                checkpointer=memory
            )
            print(" Agent initialized successfully!")
        except Exception as e:
            print(f" Error creating agent: {e}")
            self.agent_executor = None
        
        self.conversation_memory = []
    
    def run_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        
        if not self.agent_executor:
            print(" Agent not available")
            return None
        
        print(f"\n{'='*100}")
        print(f" AGENT PROCESSING: '{query}'")
        print(f"{'='*100}")
        
        try:
           
            config = {"configurable": {"thread_id": session_id}}
            result = self.agent_executor.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config
            )
            
            
            self.conversation_memory.append({
                "query": query,
                "response": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        
        except Exception as e:
            print(f" Error running query: {e}")
            return None
    
    def demonstrate_react_loop(self):
        
        print("\n" + "="*100)
        print(" THE REACT PATTERN EXPLAINED (Reasoning + Acting)")
        print("="*100)
        
        react_explanation = """
        REACT PATTERN: Reasoning → Acting → Observing → Answering
        
        
        """
        
        print(react_explanation)
    
    def test_agent_with_queries(self):
        
        
        test_queries = [
            {
                "query": "Calculate 25 * 4 for me",
                "description": "Simple math question - requires calculator"
            },
            {
                "query": "What does the word 'quantum' mean?",
                "description": "Vocabulary question - requires dictionary lookup"
            },
            {
                "query": "What is the current time and date?",
                "description": "Date/time question - requires datetime tool"
            },
            {
                "query": "Explain machine learning and calculate the square root of 144",
                "description": "Combination query - reasoning + calculation"
            },
            {
                "query": "Who is the creator of Python programming language?",
                "description": "General knowledge - agent answers directly without tools"
            }
        ]
        
        print("\n" + "="*100)
        print(" TESTING AGENT WITH 5 QUERIES")
        print("="*100)
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n{'='*100}")
            print(f"TEST {i}/{len(test_queries)}")
            print(f"Description: {test['description']}")
            print(f"{'='*100}")
            
            result = self.run_query(test["query"], session_id=f"session_{i}")
            
            if result:
                
                print(f"\n AGENT RESPONSE:")
                print("-" * 100)
                
               
                if isinstance(result, dict) and "output" in result:
                    print(result["output"])
                elif isinstance(result, dict) and "messages" in result:
                    for msg in result["messages"][-1:]:
                        if hasattr(msg, 'content'):
                            print(msg.content)
                else:
                    print(result)
    
    def print_memory_usage(self):
        
        print("\n" + "="*100)
        print("💾 CONVERSATION MEMORY ACROSS TURNS")
        print("="*100)
        
        memory_info = """
        MEMORY MANAGEMENT IN AGENTS:
        
        
        """
        
        print(memory_info)
        
        if self.conversation_memory:
            print(f"\n CURRENT CONVERSATION HISTORY: {len(self.conversation_memory)} exchanges")
            for i, exchange in enumerate(self.conversation_memory, 1):
                print(f"\n{i}. Query: {exchange['query']}")
                print(f"   Time: {exchange['timestamp']}")


def main():
    print("\n ASSIGNMENT 7: REACT AGENT WITH CUSTOM TOOLS\n")
    
    if not LANGGRAPH_AVAILABLE:
        print("  LangGraph not installed")
        print("Install with: pip install langgraph langchain langchain-ollama")
        return
    
    
    agent = ReActAgent(model_name="qwen2:0.5b")
    
   
    agent.demonstrate_react_loop()
    
   
    agent.test_agent_with_queries()
    
    
    agent.print_memory_usage()
    
    
    print("\n" + "="*100)
    
    print("="*100)
    
   
    
    print("\n" + "="*100)
    print(" REACT AGENT DEMONSTRATION COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()
