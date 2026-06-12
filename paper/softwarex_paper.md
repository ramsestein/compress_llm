# Compress LLM: A User-Centric Suite for Local Language Model Compression and Specialization

**Authors**
Ramses Marrero

**Affiliation**
Independent Researcher

**Corresponding author**
Ramses Marrero (email: ...)

---

## Abstract

Compress LLM is an open-source Python suite that enables non-technical users to compress, specialize, and serve large language models (LLMs) on consumer hardware. The software integrates state-of-the-art compression techniques---including INT8/INT4/INT2 quantization, structured and unstructured pruning, and tensor decomposition---with a comprehensive parameter-efficient fine-tuning (PEFT) module supporting LoRA, QLoRA, DoRA, MoLoRA, BitFit, IA3, and others. A novel knowledge-distillation wizard allows users to generate synthetic training data from a teacher model and train a smaller student model via a zero-code interactive CLI. The suite also exposes an Ollama-compatible REST API, enabling seamless local deployment without cloud dependencies. All functionality is accessible through interactive command-line wizards, removing the need for programming expertise. Compress LLM is released under the MIT license.

## Keywords

Large language models; model compression; quantization; pruning; LoRA; PEFT; knowledge distillation; Ollama; local AI

---

## 1. Introduction

### 1.1 Motivation

The proliferation of large language models has created a gap between cutting-edge research and practical deployment. Organizations and individual users often lack the computational resources or expertise required to fine-tune and serve billion-parameter models. Existing toolchains are fragmented, require deep familiarity with deep-learning frameworks, and rarely provide end-to-end integration from compression to local serving.

### 1.2 Novelty and contributions

Compress LLM addresses this gap by providing:

- **Ease of use**: Interactive CLI wizards guide users through compression, fine-tuning, and model serving without writing code.
- **Comprehensive compression**: A unified engine for post-training quantization, pruning, and low-rank/tensor decomposition.
- **Extensive PEFT support**: Integration of nine parameter-efficient methods, including recent variants such as DoRA and MoLoRA.
- **Knowledge distillation**: A zero-code pipeline that generates synthetic instruction datasets from a teacher model and distills knowledge into a smaller student model.
- **Local serving**: An Ollama-compatible API server for immediate deployment on consumer hardware.

### 1.3 Related work

Existing libraries such as Hugging Face PEFT, AutoGPTQ, and LLM.int8() provide individual compression or adaptation primitives, but none combine compression, fine-tuning, distillation, and local serving in a single user-centric package. Compress LLM unifies these capabilities under a consistent interactive interface.

---

## 2. Software description

### 2.1 Architecture

Compress LLM is organized into five principal modules:

1. **`create_compress`** -- Compression engine and configuration manager.
2. **`LoRa_train`** -- PEFT implementations and universal trainer.
3. **`compress_llm`** -- Public API namespace and zero-code CLI entry points.
4. **`down_report`** -- HTML/JSON/Markdown report generation with visualizations.
5. **`tests`** -- Unit and integration tests.

### 2.2 Core functionality

#### 2.2.1 Model compression

The `CompressionEngine` class applies layer-wise compression according to JSON profiles. Supported methods include:

- **Quantization**: INT8, INT4, INT2 post-training quantization.
- **Pruning**: Structured and unstructured magnitude-based pruning.
- **Decomposition**: SVD, Tucker, and MPO (matrix product operator) tensor decomposition.
- **Mixed precision**: Automatic layer-wise precision selection.

Profiles (balanced, conservative, aggressive, custom) control the trade-off between size reduction and accuracy retention.

#### 2.2.2 Parameter-efficient fine-tuning

The `PEFTUniversalTrainer` supports:

- **LoRA**, **QLoRA**, **DoRA**, **MoLoRA**
- **BitFit**, **IA3**, **Prompt Tuning**, **Adapter Tuning**, **GaLore**

Each method is exposed through a unified configuration dataclass hierarchy, allowing users to switch techniques by changing a single enum value.

#### 2.2.3 Knowledge distillation

The `KnowledgeDistiller` class:

1. Loads a large teacher model via Hugging Face Transformers.
2. Generates a synthetic CSV dataset of instruction-response pairs on a user-specified topic.
3. Fine-tunes a smaller student model using any supported PEFT method.

This pipeline is exposed through the `distill-llm` CLI wizard.

#### 2.2.4 Local serving

`ollama_compact_server.py` implements an Ollama-compatible REST API using FastAPI and Uvicorn. Models stored in the local `models/` directory are automatically discovered and exposed at `http://localhost:8000`.

### 2.3 User interface

All major features are accessible through interactive command-line wizards built with the `rich` library:

- `compress-llm` -- Interactive compression wizard.
- `finetune-llm` -- Interactive PEFT fine-tuning wizard.
- `distill-llm` -- Interactive knowledge-distillation wizard.
- `ollama-serve` -- Start the local API server.

### 2.4 Quality assurance

The project includes:

- **Linting**: `ruff`, `black`, and `mypy` configurations in `pyproject.toml`.
- **Testing**: `pytest` with coverage reporting (`pytest-cov`).
- **CI/CD**: GitHub Actions workflow running lint and tests on Python 3.9--3.12 across Ubuntu, macOS, and Windows.

---

## 3. Illustrative examples

### 3.1 Compressing a model

```bash
# Start the interactive compression wizard
python -m compress_llm.cli:main
# Or after pip install:
compress-llm
```

The wizard prompts for a model path, a compression profile, and an output directory.

### 3.2 Fine-tuning with LoRA

```bash
finetune-llm
```

The user selects LoRA, chooses a base model, picks a CSV dataset, and confirms hyper-parameters. Training proceeds with automatic progress reporting.

### 3.3 Knowledge distillation

```bash
distill-llm
```

The user specifies a topic (e.g., "medical diagnosis"), a teacher model (`microsoft/DialoGPT-medium`), and a student model (`microsoft/DialoGPT-small`). The teacher generates 100 synthetic examples, and the student is fine-tuned with LoRA.

### 3.4 Local serving

```bash
ollama-serve --models_dir ./models
```

The server is available at `http://localhost:8000` and exposes Ollama-compatible endpoints such as `/api/generate` and `/api/chat`.

---

## 4. Impact and conclusions

Compress LLM lowers the barrier to entry for model compression and specialization. By combining advanced research techniques with zero-code interactive wizards and local serving, the suite empowers non-technical users to own and operate specialized language models on consumer hardware. Future releases will extend the distillation pipeline with multi-teacher ensembles and automatic architecture search for student models.

---

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

The authors thank the Hugging Face team for the Transformers and PEFT libraries, and the open-source community for quantization and compression research.

## References

1. Dettmers, T., et al. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS*.
2. Hu, E., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
3. Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*.
4. Liu, S., et al. (2024). DoRA: Weight-decomposed low-rank adaptation. *ICML*.
5. Akiba, T., et al. (2023). MoLoRA: Mixture of LoRA experts. *arXiv preprint*.
6. Zaken, E. B., et al. (2022). BitFit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. *ACL*.
7. Liu, H., et al. (2022). Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. *NeurIPS*.
8. Hinton, G., et al. (2015). Distilling the knowledge in a neural network. *arXiv preprint*.
9. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP*.
