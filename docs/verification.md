# Verify Installation - Compress LLM

## Quick Check

Run this command to verify that everything is installed correctly:

```bash
python tests/run_all_tests.py
```

**Expected result:**
```
Tests run: 261, Errors: 0, Failures: 0
Total time: ~30 seconds
```

**If you see "Errors: 0, Failures: 0", everything is working perfectly.**

---

## Individual Checks

### 1. Check Python
```bash
python --version
# Should show: Python 3.8.x or higher
```

### 2. Check PyTorch
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Check Transformers
```bash
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### 4. Check PEFT
```bash
python -c "import peft; print(f'PEFT {peft.__version__}')"
```

---

## If There Are Problems

### Error: "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Error: "No module named 'transformers'"
```bash
pip install transformers datasets accelerate
```

### Error: "No module named 'peft'"
```bash
pip install peft
```

### Error: "CUDA not available"
- **Normal**: The system will run on CPU
- **For GPU**: Install the CUDA version of PyTorch

---

## Functionality Test

### Test 1: Compression System
```bash
python -c "
from create_compress.compression_engine import CompressionEngine
from create_compress.compression_methods import LowRankApproximation, QuantizationMethod
print('Compression system: OK')
"
```

### Test 2: LoRA System
```bash
python -c "
from LoRa_train.peft_methods import QuantizedLoRALinear
from LoRa_train.peft_methods_config import LoRAConfig, PEFTMethod
print('LoRA system: OK')
"
```

### Test 3: Dataset Manager
```bash
python -c "
from LoRa_train.dataset_manager import OptimizedDatasetManager
print('Dataset Manager: OK')
"
```

---

## System Information

### Check Available Resources:
```bash
python -c "
import psutil
import torch

print(f'Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB' if torch.cuda.is_available() else 'N/A')
"
```

---

## Complete Verification

If all tests pass, your system is ready to:

- **Compress large models**
- **Train with LoRA** and other methods
- **Optimize memory** and performance
- **Create custom models**
- **Serve models** via API

---

## Still Having Issues?

1. **Review the error** logs
2. **Verify Python** is on PATH
3. **Reinstall dependencies**: `pip install -r requirements.txt --force-reinstall`
4. **Create a new virtual environment**:
   ```bash
   python -m venv new_venv
   new_venv\Scripts\activate  # Windows
   source new_venv/bin/activate  # Linux/macOS
   pip install -r requirements.txt
   ```

---

**Your system is ready to build amazing models.**
