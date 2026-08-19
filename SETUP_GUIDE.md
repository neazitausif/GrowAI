# 🚀 Quick Setup & Installation Guide

Complete step-by-step guide to get all 8 assignments running locally.

---

## Prerequisites

- **Python 3.9+** (check with `python --version`)
- **pip** (check with `pip --version`)
- **Git** (for cloning)
- **~20 GB disk space** (for Ollama models)
- **4 GB RAM minimum** (8 GB recommended)
- **Internet connection** (for downloading models)

---

## Step 1: Setup Python Environment

### Option A: Virtual Environment 

```bash
# Create virtual environment
python -m venv venv_llm

# Activate it


# On Windows:
venv_llm\Scripts\activate
```



---

## Step 2: Install Python Dependencies

### Quick Install (All at Once)

```bash
pip install -r requirements.txt
```

### Progressive Install (If Conflicts)

```bash
# Step 1: Core packages
pip install numpy pandas requests pydantic

# Step 2: NLP packages
pip install tiktoken transformers sentence-transformers

# Step 3: Deep learning
pip install torch matplotlib

# Step 4: LLM frameworks
pip install langchain llama-index langgraph

# Step 5: Retrieval
pip install chromadb rank-bm25

# Step 6: MCP
pip install fastmcp
```

### GPU Support (Optional)

If you have NVIDIA GPU and want GPU acceleration:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For AMD GPU (ROCm):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## Step 3: Install Ollama

Ollama provides local LLM inference - essential for assignments 3-8.

### macOS/Linux

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai for manual installation

# Verify installation
ollama --version
```

### Windows

1. Download from https://ollama.ai
2. Run installer
3. Verify:
   ```bash
   ollama --version
   ```

---

## Step 4: Download Ollama Models

```bash
# Model 1: TinyLLaMA (fast, 1.1B params)
ollama pull tinyllama

# Model 2: Qwen2 0.5B (fast, 0.5B params)
ollama pull qwen2:0.5b

# Model 3: Phi3 Mini (balanced, 3.8B params)
ollama pull phi3:mini

# (Optional) Larger model for better quality
ollama pull mistral
```

**Sizes:**
- tinyllama: ~637 MB
- qwen2:0.5b: ~230 MB
- phi3:mini: ~2.2 GB
- mistral: ~4 GB

---

## Step 5: Start Ollama Server

In a **separate terminal** (keep this running):

```bash
ollama serve
```

You'll see:
```
Listening on 127.0.0.1:11434
```

✅ Server is ready when you see this message.

---

## Step 6: Verify Setup

### Test Python Packages

```python
# Run this Python code to verify all packages
python -c "
import torch
import transformers
import sentence_transformers
import langchain
import llama_index
import chromadb
import rank_bm25
print('✅ All Python packages installed!')
"
```

### Test Ollama Connection

```python
# Test connection to Ollama
python -c "
import requests
response = requests.get('http://localhost:11434/api/tags')
if response.status_code == 200:
    print(' Ollama server is running!')
    models = response.json().get('models', [])
    print(f'Available models: {len(models)}')
    for m in models:
        print(f'  - {m[\"name\"]}')
else:
    print(' Ollama server not responding')
    print('Make sure to run: ollama serve')
"
```

---

## Step 7: Run Assignments

### Run Individual Assignment

```bash
# Assignment 1: Token Explorer
python assignment_1_token_explorer.py

# Assignment 2: XOR Classifier
python assignment_2_xor_classifier.py

# Assignment 3: LLM Benchmarker
python assignment_3_llm_benchmarker.py

# ... and so on
```

### Run All Assignments in Sequence

```bash
for i in {1..8}; do
    echo "Running Assignment $i..."
    python assignment_${i}_*.py
    sleep 5
done
```

---

## Quick Reference: First Run

```bash
# Terminal 1: Start Ollama (keep running)
ollama serve

# Terminal 2: Run assignment
cd /path/to/assignments
python -m venv venv_llm
source venv_llm/bin/activate
pip install -r requirements.txt
python assignment_1_token_explorer.py
```

---

## Common Issues & Fixes

### Issue: "Ollama server not found"

```bash
# Make sure Ollama server is running in another terminal
ollama serve

# Check connection
curl http://localhost:11434/api/tags
```

### Issue: "Module not found"

```bash
# Make sure virtual environment is activated
source venv_llm/bin/activate  # macOS/Linux
venv_llm\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "Model not found"

```bash
# List available models
ollama list

# Pull missing model
ollama pull qwen2:0.5b
```

### Issue: "Out of memory"

```bash
# Use smaller model
# In script, change: model="tinyllama"

# Or reduce batch size (in code)
```

### Issue: "GPU not detected"

```bash
# Check PyTorch setup
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Port 11434 already in use

```bash
# Change Ollama port in assignment files
# In code: base_url="http://localhost:11435"

# Or kill process using port
lsof -ti:11434 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :11434   # Windows
```

---

## Development Setup (Optional)

For development and testing:

```bash
# Install development tools
pip install jupyter black pylint pytest

# Start Jupyter for interactive exploration
jupyter notebook

# Format code with Black
black assignment_1_token_explorer.py

# Lint code with Pylint
pylint assignment_1_token_explorer.py

# Run tests
pytest test_assignment_1.py
```

---





## Environment Variables (Optional)

Create `.env` file:

```bash
# Ollama settings
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_MODEL=qwen2:0.5b

# API keys (if using cloud models)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Logging
LOG_LEVEL=INFO
```

Load in code:

```python
from dotenv import load_dotenv
import os

load_dotenv()
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
```

---

## Next Steps After Installation

1.  **Verify Setup**: Run assignment 1 (no Ollama needed)
2.  **Test Ollama**: Run assignment 3 (requires Ollama)
3.  **Understand Foundations**: Assignments 1-2 before 5-8
4.  **Start Simple**: Test with tiny models before larger ones
5.  **Monitor Performance**: Watch memory and latency during runs

---

## Getting Help

### Debug Checklist

- [ ] Python 3.9+ installed?
- [ ] Virtual environment activated?
- [ ] All packages installed? (`pip list`)
- [ ] Ollama server running? (`curl http://localhost:11434/api/tags`)
- [ ] Models downloaded? (`ollama list`)
- [ ] No port conflicts? (`lsof -i :11434`)
- [ ] Sufficient disk space? (`df -h`)
- [ ] Sufficient RAM? (`free -h`)

### Useful Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "torch|langchain|ollama|transformers"

# Test imports
python -c "import torch; import langchain; print('✅')"

# Monitor Ollama
curl http://localhost:11434/api/tags

# Check disk space
df -h

# Check memory
free -h  # Linux
vm_stat  # macOS
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory  # Windows
```

---



---

## Troubleshooting Quick Links

**Still stuck?**
1. Check the README.md for detailed explanations
2. Review assignment code comments
3. Check framework documentation:
   - [LangChain](https://python.langchain.com/)
   - [LlamaIndex](https://docs.llamaindex.ai/)
   - [Ollama](https://github.com/ollama/ollama)

---

## Ready? Let's Go! 

```bash
# One final check:
python -c "print(' Everything looks good!')"

# Run your first assignment:
python assignment_1_token_explorer.py


```

---

**Next**: See README.md for detailed assignment descriptions and learning paths.
