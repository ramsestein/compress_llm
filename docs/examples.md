# Practical Examples - Compress LLM

## Common Use Cases

### Example 1: Compress a Large Model

**Scenario**: You have a 7 GB model and want to shrink it to 2 GB.

```bash
# 1. Download a sample model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model.save_pretrained('./models/dialogpt_original')
tokenizer.save_pretrained('./models/dialogpt_original')
"

# 2. Compress the model
python scripts/apply_compression.py --model_path "./models/dialogpt_original" --output_path "./models/dialogpt_compressed"

# 3. Verify the compression
python scripts/verify_compression.py --model_path "./models/dialogpt_compressed"
```

**Result**: Model reduced from ~1.5 GB to ~400 MB.

---

### Example 2: Improve a Model with LoRA

**Scenario**: You want the model to perform better in Spanish.

```bash
# 1. Prepare Spanish data (CSV file)
# Create file: datasets/spanish.csv
# Format: text,response
# Example:
# "Hola, como estas?", "Hola! Estoy muy bien, y tu?"

# 2. Run fine-tuning
python scripts/finetune_peft.py

# 3. Follow the on-screen instructions:
# - Select: LoRA
# - Model: microsoft/DialoGPT-small
# - Data: datasets/spanish.csv
# - Epochs: 3
# - Learning rate: 2e-4
```

**Result**: Spanish-tuned model with only ~10 MB of additional parameters.

---

### Example 3: Optimize for Low Memory

**Scenario**: You only have 8 GB of RAM.

```bash
# 1. Use QLoRA (quantization + LoRA)
python scripts/finetune_peft.py

# 2. Recommended configuration:
# - Method: QLoRA
# - Bits: 4
# - Rank: 8
# - Batch size: 1
# - Gradient accumulation: 4
```

**Result**: Training feasible on 8 GB RAM.

---

### Example 4: Create a Custom Chatbot

**Scenario**: You want a chatbot for your business.

```bash
# 1. Prepare conversation data
# Create: datasets/business_chatbot.csv
# Format:
# "What are your opening hours?", "Our hours are 9:00 to 18:00"
# "Do you offer free shipping?", "Yes, on orders over $50"

# 2. Train with IA3 (very fast)
python scripts/finetune_peft.py
# - Method: IA3
# - Epochs: 1
# - Learning rate: 1e-3

# 3. Test the model
python tests/test_compressed_model.py --model_path "./finetuned_models/my_chatbot"
```

**Result**: Custom chatbot ready in ~30 minutes.

---

### Example 5: Model Server

**Scenario**: You want to serve models through an API.

```bash
# 1. Start the server
python scripts/ollama_compact_server.py --models_dir "./models"

# 2. The server will be available at:
# http://localhost:8000

# 3. Use with curl:
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"model": "dialogpt_compressed", "prompt": "Hello, how are you?"}'
```

**Result**: REST API for your local models.

---

## Method Comparison

| Method | Speed | Memory | Quality | Recommended Use |
|--------|-------|--------|---------|-----------------|
| **LoRA** | *** | **** | **** | General purpose |
| **IA3** | ***** | ***** | *** | Quick experiments |
| **BitFit** | ***** | ***** | ** | Very limited resources |
| **QLoRA** | ** | ***** | **** | Low-memory training |
| **DoRA** | ** | *** | ***** | Maximum quality |

---

## Recommended Configurations

### Beginners:
```bash
python scripts/finetune_peft.py
# - Method: LoRA
# - Rank: 16
# - Alpha: 32
# - Epochs: 3
# - Learning rate: 2e-4
```

### Intermediate:
```bash
python scripts/finetune_peft.py
# - Method: QLoRA
# - Bits: 4
# - Rank: 32
# - Epochs: 5
# - Learning rate: 1e-4
```

### Advanced:
```bash
python scripts/finetune_peft.py
# - Method: DoRA
# - Rank: 64
# - Alpha: 128
# - Epochs: 10
# - Learning rate: 5e-5
```

---

## Progress Monitoring

### During Training:
- **Loss**: Should decrease (e.g., 2.5 -> 1.2)
- **Accuracy**: Should increase (e.g., 0.3 -> 0.8)
- **Memory**: Must not exceed available RAM

### Expected Metrics:
- **LoRA**: Loss < 1.5 after 3 epochs
- **IA3**: Loss < 2.0 after 1 epoch
- **QLoRA**: Loss < 1.8 after 5 epochs

---

## Common Problems and Solutions

### Error: "CUDA out of memory"
```bash
# Solution: Reduce batch size
python scripts/finetune_peft.py
# - Batch size: 1
# - Gradient accumulation: 8
```

### Error: "Model too large"
```bash
# Solution: Compress first
python scripts/apply_compression.py --model_path "large_model" --output_path "small_model"
```

### Error: "No convergence"
```bash
# Solution: Adjust learning rate
python scripts/finetune_peft.py
# - Learning rate: 1e-5 (lower)
# - Epochs: 10 (more time)
```

---

## Success!

When you see these messages, everything is working:

```
Model compressed successfully
Training completed
Model saved in: ./finetuned_models/
Final loss: 1.234
Final accuracy: 0.856
```

**Your model is ready to use.**
