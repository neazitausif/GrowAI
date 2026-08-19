# GROWAI LLM Engineering Masterclass - Complete Solutions

 **Complete, production-ready solutions for all 8 LLM Engineering assignments**

This repository contains fully implemented solutions for the GROWAI LLM Engineering Masterclass covering Modules 1-7, with:
-  Complete, commented Python code
-  Error handling and validation
-  Example usage and test cases
-  Detailed explanations for video walkthroughs
-  Production best practices

---

##  Assignments Overview

### **Assignment 1: Token Explorer & Semantic Similarity Engine** 
**Modules 1-2: Foundations**
- **Focus**: Tokenization and embeddings
- **Key Concepts**: 
  - TikToken (GPT-style) vs HuggingFace tokenizers
  - Token counting and comparison
  - Sentence embeddings with SentenceTransformers
  - Cosine similarity ranking
- **File**: `assignment_1_token_explorer.py`
- **Key Takeaway**: Understand how LLMs see text and meaning

```bash
# Install dependencies
pip install tiktoken transformers sentence-transformers scikit-learn

# Run
python assignment_1_token_explorer.py
```

**Output**: Side-by-side token comparison + semantic similarity rankings with 3 test cases (English, English, Chinese)

---

### **Assignment 2: Neural Network from Scratch - XOR Classifier**
**Modules 1-2: Python & PyTorch**
- **Focus**: Manual backpropagation and why depth matters
- **Key Concepts**:
  - 2-layer neural network architecture
  - Manual training loop with forward/backward pass
  - Why single-layer perceptron fails on XOR
  - Non-linear decision boundaries
- **File**: `assignment_2_xor_classifier.py`
- **Key Takeaway**: Deep learning foundations - why networks need multiple layers

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run
python assignment_2_xor_classifier.py
```

**Output**: 
- Training convergence logs (every 500 epochs)
- Perfect accuracy on all 4 XOR cases
- Loss convergence plot visualization
- Detailed explanation of why depth matters

---

### **Assignment 3: Local LLM Benchmarker**
**Module 3: Open-Source LLMs**
- **Focus**: Model selection and production tradeoffs
- **Key Concepts**:
  - Running multiple Ollama models
  - Response time measurement
  - Quality scoring and ranking
  - CSV reporting
  - Speed vs quality tradeoff analysis
- **File**: `assignment_3_llm_benchmarker.py`
- **Key Takeaway**: How to evaluate and select models for production

```bash
# Prerequisites: Ollama running
ollama serve

# In another terminal:
ollama pull tinyllama
ollama pull qwen2:0.5b
ollama pull phi3:mini

# Install dependencies
pip install requests pandas

# Run
python assignment_3_llm_benchmarker.py
```

**Output**:
- Benchmark results for 3 models × 5 prompts
- Comparison table by model
- Comparison by task category
- CSV export of all results
- Production recommendations

---

### **Assignment 4: Prompt Engineering Showdown**
**Module 4: Prompt Engineering**
- **Focus**: Zero-shot vs Few-shot vs Chain-of-thought
- **Key Concepts**:
  - Zero-shot prompting baseline
  - Few-shot examples improving quality
  - Chain-of-thought reasoning
  - Prompt failure modes
  - Structured output with JSON
- **File**: `assignment_4_prompt_engineering.py`
- **Key Takeaway**: Prompt quality is the most cost-effective LLM lever

```bash
# Prerequisites: Ollama running with qwen2:0.5b
ollama pull qwen2:0.5b

# Install dependencies
pip install requests pydantic

# Run
python assignment_4_prompt_engineering.py
```

**Output**:
- Side-by-side comparison of 3 techniques on 3 tasks
- Failure mode demonstration
- Fixed structured output with JSON parsing
- Detailed comparison showing quality improvements

---

### **Assignment 5: Multi-Step Content Pipeline**
**Module 4: LangChain & LlamaIndex**
- **Focus**: Multi-step workflows and framework comparison
- **Key Concepts**:
  - LangChain LCEL (pipe operator)
  - LlamaIndex query engine pattern
  - Chaining multiple LLM calls
  - Output dependency management
- **File**: `assignment_5_content_pipeline.py`
- **Key Takeaway**: Framework choice matters - understand tradeoffs

```bash
# Prerequisites: Ollama running
ollama pull qwen2:0.5b

# Install dependencies
pip install langchain langchain-ollama llama-index llama-index-llms-ollama

# Run
python assignment_5_content_pipeline.py
```

**Output**:
- LangChain pipeline: explanation → quiz → answers
- LlamaIndex pipeline: same workflow, different syntax
- Framework comparison table
- Decision framework for when to use each

---

### **Assignment 6: Document Q&A System with Hybrid Search**
**Module 5: RAG Engineering**
- **Focus**: Complete RAG pipeline with multiple retrieval strategies
- **Key Concepts**:
  - Document chunking with overlap
  - Vector embeddings (ChromaDB)
  - BM25 keyword search
  - Reciprocal Rank Fusion (hybrid search)
  - Cross-encoder reranking
  - LLM answer generation with grounding
- **File**: `assignment_6_rag_system.py`
- **Key Takeaway**: RAG is the most deployed LLM pattern in production

```bash
# Install dependencies
pip install sentence-transformers rank-bm25 chromadb requests

# Run
python assignment_6_rag_system.py
```

**Output**:
- Document ingestion (2-page sample on photosynthesis)
- Vector + BM25 indices created
- 5 test queries with hybrid search results
- Ranked top-3 results from each
- Cross-encoder reranking visualization
- RAG architecture diagram and best practices

---

### **Assignment 7: ReAct Agent with Custom Tools**
**Module 7: AI Agents**
- **Focus**: ReAct pattern and tool-using agents
- **Key Concepts**:
  - Reasoning + Acting + Observing + Answering
  - Custom tool implementation
  - LangGraph agent creation
  - Memory management with MemorySaver
  - Agent decision-making
- **File**: `assignment_7_react_agent.py`
- **Key Takeaway**: Agents represent evolution from pipelines to decision-makers

```bash
# Prerequisites: Ollama running
ollama pull qwen2:0.5b

# Install dependencies
pip install langgraph langchain langchain-ollama

# Run
python assignment_7_react_agent.py
```

**Output**:
- ReAct loop visualization
- 5 test queries demonstrating agent reasoning
- Tool selection decisions (calculator, dictionary, datetime)
- Memory management across conversation turns
- Production implementation patterns

---

### **Assignment 8: Multi-Agent Research Assistant & MCP Server**
**Module 7: Multi-Agent Systems & MCP**
- **Focus**: Scalable agent architectures and MCP protocol
- **Key Concepts**:
  - Supervisor + Worker agent pattern
  - Specialist agents with focused tools
  - Query routing mechanisms
  - MCP (Model Context Protocol) basics
  - Inter-agent communication
- **File**: `assignment_8_multi_agent_mcp.py`
- **Key Takeaway**: Production AI systems use specialized agents, not monoliths

```bash
# Prerequisites: Ollama running
ollama pull qwen2:0.5b

# Install dependencies
pip install langgraph langchain langchain-ollama fastmcp

# Run
python assignment_8_multi_agent_mcp.py
```

**Output**:
- Multi-agent workflow visualization
- Supervisor routing decisions
- Worker agent specialization
- MCP server concepts and architecture
- 4 test queries demonstrating routing
- Production deployment patterns

---

## 🚀 Quick Start

### Prerequisites
1. **Python 3.9+**
2. **Ollama** (for LLM inference)
   ```bash
   # Install from https://ollama.ai
   # Start Ollama server
   ollama serve
   ```

3. **Python packages**
   ```bash
   # Core dependencies
   pip install requests numpy pandas

   # Deep learning
   pip install torch

   # NLP & Embeddings
   pip install transformers sentence-transformers

   # LLM Frameworks
   pip install langchain langchain-ollama langgraph
   pip install llama-index llama-index-llms-ollama

   # Retrieval & Search
   pip install rank-bm25 chromadb scikit-learn

   # MCP
   pip install fastmcp

   # Visualization
   pip install matplotlib
   ```

### Running All Assignments

```bash
# Each assignment is standalone
python assignment_1_token_explorer.py
python assignment_2_xor_classifier.py
python assignment_3_llm_benchmarker.py
python assignment_4_prompt_engineering.py
python assignment_5_content_pipeline.py
python assignment_6_rag_system.py
python assignment_7_react_agent.py
python assignment_8_multi_agent_mcp.py
```

---



##  Key Concepts Across Assignments

### Tokenization & Embeddings (Assignments 1)
- How LLMs see text as tokens
- Semantic similarity through embeddings
- Token counting for cost/performance

### Deep Learning Fundamentals (Assignment 2)
- Backpropagation and gradient descent
- Why networks need depth
- Neural network architecture design

### Model Selection (Assignment 3)
- Speed vs quality tradeoffs
- Latency and throughput measurement
- Production benchmarking

### Prompt Engineering (Assignment 4)
- Iterative improvement: zero → few → chain-of-thought
- Structured output extraction
- Debugging prompt failures

### System Design (Assignments 5-8)
- **Pipelines**: Sequential LLM calls (Assignment 5)
- **Retrieval**: Hybrid search and reranking (Assignment 6)
- **Agents**: Tool-using decision makers (Assignment 7)
- **Scaling**: Multi-agent architectures (Assignment 8)

---


