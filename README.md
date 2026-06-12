# Compress LLM - Language Model Compression and Fine-Tuning Suite

A comprehensive system for compressing and improving language models using advanced compression and parameter-efficient fine-tuning techniques.

## What does this project do?

This project enables you to:

- **Compress large models** to reduce disk footprint and improve inference speed
- **Fine-tune models** with techniques such as LoRA, IA3, BitFit, and more
- **Optimize memory usage** on consumer hardware
- **Create specialized models** tailored to domain-specific tasks

## System Requirements

### Minimum:
- **Windows 10/11**, **Linux**, or **macOS**
- **Python 3.8** or later
- **8 GB RAM** (16 GB recommended)
- **10 GB free disk** space

### Recommended:
- **NVIDIA GPU** with 8+ GB VRAM (for faster training)
- **32 GB RAM** (for large models)
- **100 GB free disk** space (for multiple models)

## Installation

### 1. Download the project

```bash
# Option 1: Clone from Git
git clone https://github.com/tu-usuario/compress_llm.git
cd compress_llm

# Option 2: Download ZIP
# Download the ZIP archive from GitHub and extract it
```

### 2. Install dependencies

Open a terminal in the project folder and run:

```bash
# Automatic installation
python scripts/install.py
```

**Or manually:**

```bash
# Create a virtual environment
python -m venv comp_venv

# Activate the environment
# Windows:
comp_venv\Scripts\activate
# Linux/macOS:
source comp_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Interactive Interface (recommended for beginners)

```bash
# Run the interactive assistant
python scripts/finetune_peft.py
```

This guides you step-by-step to:
- Select a model
- Choose a fine-tuning method
- Configure hyper-parameters
- Start training

### Option 2: Standalone Scripts

#### Compress a model:

```bash
python scripts/apply_compression.py --model_path "path/to/model" --output_path "compressed_model"
```

#### Fine-tune with LoRA:

```bash
python scripts/finetune_lora.py
```

#### Verify a model:

```bash
python tests/test_compressed_model.py --model_path "path/to/model"
```

## Available Methods

### Compression:
- **INT8/INT4/INT2**: Post-training quantization
- **Pruning**: Structured and unstructured weight pruning
- **SVD/Tucker/MPO**: Low-rank and tensor decomposition
- **Mixed Precision**: Layer-wise precision selection

### Fine-Tuning:
- **LoRA**: Low-rank adaptation
- **IA3**: Infused adapter by inhibiting and amplifying inner activations
- **BitFit**: Bias-term fine-tuning
- **Adapter**: Bottleneck adapter layers
- **QLoRA**: Quantized LoRA
- **DoRA**: Weight-decomposed low-rank adaptation
- **MoLoRA**: Mixture of LoRA experts
- **Compacter/KronA/S4/Houlsby**: Advanced adaptation methods

## Project Structure

```
compress_llm/
├── compress_llm/           # Core package (CLI, distillation)
├── create_compress/        # Compression engine
├── LoRa_train/             # Fine-tuning engine
├── down_report/            # Reporting utilities
├── scripts/                # Standalone utilities
│   ├── install.py
│   ├── finetune_peft.py
│   └── ...
├── tests/                  # Test suite
├── docs/                   # Additional documentation
├── paper/                  # SoftwareX paper draft
├── datasets/               # Training data
├── models/                 # Saved models
├── compression_analysis/   # Compression reports
├── finetuned_models/       # Fine-tuned outputs
├── requirements.txt        # Dependencies
└── pyproject.toml          # Package metadata
```

## Usage Examples

### Example 1: Compress a model

```bash
# 1. Download a sample model (DialoGPT)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small').save_pretrained('./models/my_model')"

# 2. Compress it
python scripts/apply_compression.py --model_path "./models/my_model" --output_path "./models/my_model_compressed"
```

### Example 2: Fine-tune with LoRA

```bash
# 1. Prepare training data
# Place your .csv files in the datasets/ folder

# 2. Run fine-tuning
python scripts/finetune_peft.py
# Follow the on-screen instructions
```

### Example 3: Ollama-compatible server

```bash
# Start an Ollama-compatible API server
python scripts/ollama_compact_server.py --models_dir "./models"
```

## Troubleshooting

### Error: "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Error: "CUDA out of memory"
- Reduce batch size
- Use a smaller model
- Close other GPU applications

### Error: "Not enough disk space"
- Free disk space
- Use more aggressive compression
- Remove temporary models

### Error: "Python not found"
- Ensure Python is on your PATH
- Reinstall Python and select "Add to PATH"

## Monitoring and Reports

The system automatically generates:

- **Compression reports**: Detailed compression analysis
- **Training metrics**: Loss, accuracy, etc.
- **Memory usage**: RAM and VRAM statistics
- **Execution times**: Duration of each process

Reports are saved in:
- `compression_analysis/` for compression results
- `finetuned_models/` for training results

## Security and Privacy

- **Local processing**: Everything runs on your machine
- **Offline**: No data is sent to external servers
- **Own models**: Use your own models and data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## Support

If you encounter issues:

1. Review this documentation
2. Run the tests: `python -m unittest discover tests` or `python tests/run_all_tests.py`
3. Search existing GitHub Issues
4. Open a new Issue with details

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Microsoft for DialoGPT
- The open-source community

---

**Compress and specialize your language models with ease.**
