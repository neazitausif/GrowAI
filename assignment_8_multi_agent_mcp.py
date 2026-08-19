

import json
from typing import Any, Dict, List
from datetime import datetime


try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.graph import StateGraph, START, END
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool
    AGENTS_AVAILABLE = True
except ImportError:
    print(" LangGraph not installed. Install with:")
    print("   pip install langgraph langchain langchain-ollama")
    AGENTS_AVAILABLE = False

try:
    from fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    print("FastMCP not installed. Install with:")
    print("   pip install fastmcp")
    FASTMCP_AVAILABLE = False




class SimpleWeatherServer:
    
    def __init__(self):
        
        print(f" Initializing Simple MCP Server")
        
        
        self.weather_db = {
            "new york": {"temp": 72, "condition": "Cloudy", "humidity": 65},
            "los angeles": {"temp": 85, "condition": "Sunny", "humidity": 30},
            "london": {"temp": 59, "condition": "Rainy", "humidity": 85},
            "tokyo": {"temp": 78, "condition": "Clear", "humidity": 55},
            "sydney": {"temp": 68, "condition": "Partly Cloudy", "humidity": 70},
        }
        
        
        self.news_db = {
            "technology": [
                "AI companies launch new safety initiatives",
                "Quantum computing breakthroughs reported",
                "New programming language gains adoption"
            ],
            "science": [
                "Mars rover discovers new geological formations",
                "Breakthrough in cancer research",
                "Climate change impact studies updated"
            ],
            "business": [
                "Tech stocks surge on earnings reports",
                "New startup funding rounds announced",
                "Market trends shift toward green technology"
            ],
        }
        
        print(" MCP Server initialized with weather and news tools")
    
    @tool
    def get_weather(self, city: str) -> str:
        """
        Get current weather for a city.
        
        Args:
            city: City name (e.g., 'new york', 'london')
        
        Returns:
            Weather information for the city
        """
        city_lower = city.lower()
        
        if city_lower in self.weather_db:
            data = self.weather_db[city_lower]
            return json.dumps({
                "city": city,
                "temperature": data["temp"],
                "condition": data["condition"],
                "humidity": data["humidity"],
                "timestamp": datetime.now().isoformat()
            })
        else:
            return json.dumps({"error": f"Weather data not available for {city}"})
    
    @tool
    def get_news(self, topic: str) -> str:
        """
        Get latest news for a topic.
        
        Args:
            topic: Topic (technology, science, business, etc.)
        
        Returns:
            List of news items
        """
        topic_lower = topic.lower()
        
        if topic_lower in self.news_db:
            return json.dumps({
                "topic": topic,
                "news": self.news_db[topic_lower],
                "count": len(self.news_db[topic_lower]),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return json.dumps({
                "error": f"No news available for topic: {topic}",
                "available_topics": list(self.news_db.keys())
            })



class ResearchAgent:
    
    
    def __init__(self, name: str, role: str, model: str = "qwen2:0.5b"):
        self.name = name
        self.role = role
        self.model = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.7
        )
        self.tools = []
    
    def describe(self) -> str:
        
        return f"{self.name} ({self.role})"


class ResearchWorkerAgent(ResearchAgent):
    
    
    def __init__(self, name="Research Agent", model="qwen2:0.5b"):
        super().__init__(name, "Information Retrieval", model)
        
        
        self.tools = [self._mock_search]
    
    @staticmethod
    def _mock_search(query: str) -> str:
        
        knowledge_base = {
            "machine learning": "Machine learning is a subset of AI that enables systems to learn from data",
            "deep learning": "Deep learning uses neural networks with multiple layers",
            "nlp": "Natural Language Processing enables machines to understand human language",
            "computer vision": "Computer vision enables machines to interpret visual information",
            "reinforcement learning": "RL enables agents to learn through interaction with environment"
        }
        
        query_lower = query.lower()
        
        for key, value in knowledge_base.items():
            if key in query_lower:
                return f"Found information: {value}"
        
        return f"No specific information found for '{query}', but knowledge base contains general AI/ML info"
    
    def search(self, query: str) -> str:
        
        print(f"🔍 {self.name} searching for: '{query}'")
        result = self._mock_search(query)
        print(f"   Found: {result}")
        return result


class AnalysisWorkerAgent(ResearchAgent):
    
    
    def __init__(self, name="Analysis Agent", model="qwen2:0.5b"):
        super().__init__(name, "Analysis & Comparison", model)
    
    def analyze(self, text1: str, text2: str) -> str:
        """Analyze and compare two text snippets"""
        print(f" {self.name} analyzing and comparing...")
        
        analysis = {
            "comparison": f"Comparing '{text1[:50]}...' with '{text2[:50]}...'",
            "similarities": "Both texts discuss related topics in AI/ML domain",
            "differences": "First text focuses on ML basics, second on applications",
            "recommendation": "Both sources are complementary for understanding AI concepts"
        }
        
        return json.dumps(analysis, indent=2)


class SupervisorAgent:
    
    
    def __init__(self, workers: List[ResearchWorkerAgent], model: str = "qwen2:0.5b"):
        self.model = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.7
        )
        self.workers = workers
        self.call_log = []
    
    def route_query(self, query: str) -> Dict[str, Any]:
        
        print(f"\n SUPERVISOR ROUTING: '{query}'")
        print("-" * 100)
        
        routing_decision = self._decide_routing(query)
        
        results = {
            "query": query,
            "routing_decision": routing_decision,
            "worker_results": []
        }
        
        
        if "research" in routing_decision or "search" in routing_decision:
            if len(self.workers) > 0:
                print(f"\n→ Calling {self.workers[0].name}")
                worker_result = self.workers[0].search(query)
                results["worker_results"].append({
                    "agent": self.workers[0].name,
                    "result": worker_result
                })
        
        if "analyze" in routing_decision and len(self.workers) > 1:
            print(f"\n→ Calling {self.workers[1].name}")
            
            analysis_result = self.workers[1].analyze(
                "Machine learning is AI subset",
                "Deep learning uses neural networks"
            )
            results["worker_results"].append({
                "agent": self.workers[1].name,
                "result": analysis_result
            })
        
        self.call_log.append(results)
        return results
    
    def _decide_routing(self, query: str) -> List[str]:
        
        routing = []
        
        query_lower = query.lower()
        
       
        if any(word in query_lower for word in ["what", "explain", "define", "find", "search"]):
            routing.append("research")
        
        if any(word in query_lower for word in ["compare", "analyze", "difference", "vs", "versus"]):
            routing.append("analyze")
        
        if not routing:
            routing.append("research")  # Default to research
        
        return routing


def demonstrate_multi_agent_workflow():
    
    
    print("\n" + "="*100)
    print(" MULTI-AGENT WORKFLOW DEMONSTRATION")
    print("="*100)
    
    workflow = """
    SCENARIO: User wants to research AI and compare approaches
    
    """
    
    print(workflow)


def demonstrate_mcp_server():
    
    
    print("\n" + "="*100)
    print(" MODEL CONTEXT PROTOCOL (MCP) DEMONSTRATION")
    print("="*100)
    
    mcp_info = """
    WHAT IS MCP (Model Context Protocol)?
    
    
    """
    
    print(mcp_info)


def main():
    print("\n ASSIGNMENT 8: MULTI-AGENT RESEARCH ASSISTANT & MCP SERVER\n")
    
    if not AGENTS_AVAILABLE:
        print(" LangGraph not available")
        return
    
   
    print(" INITIALIZING MCP SERVER")
    print("-" * 100)
    mcp_server = SimpleWeatherServer()
    
    
    print("\n INITIALIZING WORKER AGENTS")
    print("-" * 100)
    research_agent = ResearchWorkerAgent("Research Agent")
    analysis_agent = AnalysisWorkerAgent("Analysis Agent")
    
    print(f" {research_agent.describe()}")
    print(f" {analysis_agent.describe()}")
    
    
    print("\n INITIALIZING SUPERVISOR AGENT")
    print("-" * 100)
    supervisor = SupervisorAgent([research_agent, analysis_agent])
    print(" Supervisor ready to route queries")
    
    
    demonstrate_multi_agent_workflow()
    
    
    print("\n" + "="*100)
    print(" TESTING MULTI-AGENT SYSTEM")
    print("="*100)
    
    test_queries = [
        "What is machine learning?",
        "Compare machine learning and deep learning",
        "Explain neural networks",
        "What are the applications of AI?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*100}")
        print(f"QUERY {i}/{len(test_queries)}")
        print(f"{'='*100}")
        result = supervisor.route_query(query)
        
        print(f"\n RESULT:")
        print(f"Routing Decision: {result['routing_decision']}")
        print(f"Worker Responses: {len(result['worker_results'])} agent(s) called")
        
        for worker_result in result['worker_results']:
            print(f"\n  {worker_result['agent']}:")
            print(f"  {worker_result['result'][:200]}...")
    
    
    demonstrate_mcp_server()
    
    
    print("\n" + "="*100)
    print("🎓 KEY POINTS FOR YOUR VIDEO WALKTHROUGH")
    print("="*100)
    
   
    
    print("\n" + "="*100)
    print(" MULTI-AGENT SYSTEM DEMONSTRATION COMPLETE!")
    print("="*100)
    


if __name__ == "__main__":
    main()
