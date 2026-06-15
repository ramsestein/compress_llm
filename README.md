# Compress LLM: A Comprehensive Suite for LLM Compression and Parameter-Efficient Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-262%20passing-brightgreen.svg)](https://github.com/ramsestein/compress_llm/tree/main/tests)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

> **Compress LLM** is an open-source toolkit that enables researchers and practitioners to compress and fine-tune large language models on consumer hardware using a unified, extensible Python API and CLI.

---

## Table of Contents

- [Overview](#overview)
- [Statement of Need](#statement-of-need)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Functionality](#core-functionality)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

Compress LLM provides a single, coherent framework for:

1. **Model Compression** — reducing model size and accelerating inference via quantization, pruning, and low-rank/tensor decomposition.
2. **Parameter-Efficient Fine-Tuning (PEFT)** — adapting pre-trained models to downstream tasks with minimal trainable parameters (LoRA, IA\u00b3, QLoRA, DoRA, and more).
3. **Deployment** — serving compressed models through an Ollama-compatible REST API for rapid prototyping.

All processing is performed locally; no data leaves the host machine.

---

## Statement of Need

Running state-of-the-art LLMs on consumer GPUs or CPU-only workstations is increasingly important for reproducible research, privacy-sensitive applications, and resource-constrained environments. Existing tools often address compression or fine-tuning in isolation, require deep framework expertise, or depend on cloud APIs.

Compress LLM unifies both workflows in a single, well-tested Python package with:

- **Automatic compression profiling** that selects layer-wise methods based on target size or speed budgets.
- **Extensive PEFT support** covering both mainstream and recent methods (BitFit, Compacter, KronA, S4 adapters, Houlsby, MoLoRA).
- **A local Ollama-compatible server** for immediate integration with existing front-ends.
- **Comprehensive unit and end-to-end tests** (262 tests, continuously verified).

---

## System Requirements

| Level | OS | Python | RAM | Disk | GPU |
|---|---|---|---|---|---|
| **Minimum** | Windows 10/11, Linux, macOS | 3.10 | 8 GB | 10 GB | Optional |
| **Recommended** | Windows 10/11, Linux | 3.10+ | 32 GB | 100 GB | NVIDIA with 8 GB+ VRAM |

Optional dependencies:
- `tensorly` — for Matrix-Product-Operator (MPO) decomposition.
- `flash-attn` — for memory-efficient attention during fine-tuning.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ramsestein/compress_llm.git
cd compress_llm
```

### 2. Install dependencies

**Automated (recommended):**

```bash
python scripts/install.py
```

**Manual (virtual environment):**

```bash
python -m venv comp_venv
# Windows:
comp_venv\Scripts\activate
# Linux / macOS:
# source comp_venv/bin/activate

pip install -r requirements.txt
```

---

## Quick Start

### Compress a model

```bash
# 1. Download a sample model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small').save_pretrained('./models/dialoGPT')"

# 2. Compress it automatically
python scripts/apply_compression.py dialoGPT --models-dir ./models --force
```

### Fine-tune with an interactive assistant

```bash
python scripts/finetune_peft.py
```

Follow the on-screen prompts to select a model, method, and hyper-parameters.

### Serve a compressed model locally

```bash
python scripts/ollama_compact_server.py --models_dir ./models
```

The server exposes an Ollama-compatible API at `http://localhost:11434`.

---

## Core Functionality

### Compression methods

| Method | Type | Description |
|---|---|---|
| `int8_quantization` | Quantization | 8-bit post-training quantization |
| `int4_quantization` | Quantization | 4-bit post-training quantization |
| `magnitude_pruning` | Pruning | Unstructured weight pruning by magnitude |
| `structured_pruning` | Pruning | Structured (row/column) pruning |
| `svd` | Decomposition | Singular-value decomposition |
| `tucker_decomposition` | Decomposition | Tucker tensor decomposition |
| `mpo` | Decomposition | Matrix-product operator (requires `tensorly`) |

### PEFT methods

| Method | Description |
|---|---|
| LoRA | Low-rank adaptation |
| IA³ | Infused adapter by inhibiting and amplifying inner activations |
| QLoRA | Quantized LoRA with 4-bit base weights |
| DoRA | Weight-decomposed low-rank adaptation |
| MoLoRA | Mixture of LoRA experts |
| BitFit | Bias-term fine-tuning only |
| Adapter | Bottleneck adapter layers |
| Compacter | Hyper-complex adapter layers |
| KronA | Kronecker-factorized adapters |
| S4 | Structured state-space adapters |
| Houlsby | Parallel & serial adapter configuration |
| GaLore | Gradient low-rank projection |

---

## Testing

Run the full test suite with Python's built-in `unittest`:

```bash
python -m unittest discover tests
```

Or use the custom runner for categorized output:

```bash
python tests/run_all_tests.py
```

Expected result:
- **262 tests** executed
- **100 % pass rate** (10 skipped when optional dependencies are absent)
- Typical runtime: ~3-5 seconds on CPU

To run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

---

## Continuous Integration

The project is configured for GitHub Actions workflows:

- **Lint** — `ruff check .` on every push / PR.
- **Test** — `python -m unittest discover tests` on Python 3.10, 3.11, 3.12.
- **E2E** — end-to-end validation with `microsoft/DialoGPT-small`.

See `.github/workflows/ci.yml` for the full pipeline definition.

---

## Project Structure

```
compress_llm/
├── compress_llm/             # Core package (CLI entry points, distillation)
├── create_compress/          # Compression engine & configuration manager
├── LoRa_train/               # PEFT fine-tuning engine
├── scripts/                  # Standalone CLI utilities
│   ├── apply_compression.py
│   ├── finetune_peft.py
│   └── ollama_compact_server.py
├── tests/                    # Test suite (262 tests)
├── docs/                     # Additional documentation
│   ├── CONTRIBUTING.md
│   ├── CHANGELOG.md
│   ├── verification.md
│   └── project_summary.md
├── paper/                    # SoftwareX manuscript
├── compression_analysis/     # Generated compression reports
├── datasets/                 # Training datasets
├── models/                   # Saved model checkpoints
├── requirements.txt          # Runtime dependencies
└── pyproject.toml            # Package metadata & ruff configuration
```

---

## Documentation

- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — developer setup, coding standards, and PR guidelines.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — release history and breaking changes.
- [`docs/verification.md`](docs/verification.md) — step-by-step system verification checklist.
- [`docs/project_summary.md`](docs/project_summary.md) — executive summary for reviewers.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Ensure all tests pass: `python -m unittest discover tests`.
4. Run linting and formatting: `ruff format . && ruff check .`.
5. Commit your changes.
6. Open a Pull Request against `main`.

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the full contributor guide.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{marrero_compress_llm_2026,
  author = {Marrero, Ramses},
  title = {Compress LLM: A Comprehensive Suite for LLM Compression and Parameter-Efficient Fine-Tuning},
  url = {https://github.com/ramsestein/compress_llm},
  version = {1.0.0},
  date = {2026-06-11},
  license = {MIT}
}
```

A `CITATION.cff` file is also provided in the repository root for automated citation parsers.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library.
- [Microsoft](https://github.com/microsoft/DialoGPT) for the DialoGPT model family.
- The open-source community for invaluable feedback and contributions.

---

**Compress and specialize your language models with ease.**
