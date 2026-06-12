# Quick Installation - Compress LLM

## Installation in 3 Steps

### Step 1: Download
```bash
git clone https://github.com/tu-usuario/compress_llm.git
cd compress_llm
```

### Step 2: Install
```bash
python scripts/install.py
```

### Step 3: Run
```bash
python scripts/finetune_peft.py
```

Done!

---

## Troubleshooting

### Check Python:
```bash
python --version
# Must show Python 3.8 or higher
```

### Install manually:
```bash
pip install torch transformers datasets accelerate peft bitsandbytes
```

### Activate virtual environment:
```bash
python -m venv comp_venv
comp_venv\Scripts\activate  # Windows
source comp_venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

---

## Still having issues?

1. **Python error**: Install Python from [python.org](https://python.org)
2. **Memory error**: Close other applications
3. **Network error**: Check your internet connection

---

**The system is ready to use.**
