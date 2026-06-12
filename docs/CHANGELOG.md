# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-12

### Added

- Comprehensive model compression engine supporting INT8/INT4/INT2 quantization, structured and unstructured pruning, SVD/Tucker/MPO tensor decomposition, and mixed-precision layer-wise selection.
- Parameter-efficient fine-tuning (PEFT) suite with LoRA, QLoRA, DoRA, MoLoRA, BitFit, IA3, Prompt Tuning, Adapter Tuning, GaLore, Compacter, KronA, S4, and Houlsby methods.
- Interactive zero-code CLI wizard (`finetune_peft.py`) for non-technical users.
- Ollama-compatible REST API server (`ollama_compact_server.py`) for local model serving.
- Compression configuration manager with JSON-based profiles (balanced, conservative, aggressive, custom).
- Dataset manager supporting CSV, JSON, Parquet, and Hugging Face `datasets` formats.
- HTML/JSON/Markdown report generator with visualization of compression metrics and layer statistics.
- `pyproject.toml` with modern Python packaging, `ruff` linting/formatting, and `mypy` tooling.
- Academic metadata: `CITATION.cff`, `AUTHORS.md`, and `CHANGELOG.md`.
- MIT license.

### Changed

- Translated all user-facing documentation, CLI prompts, log messages, and docstrings from Spanish to English.
- Removed emojis and informal language to adopt a neutral academic tone.
- Restructured project for proper Python packaging with `compress_llm` namespace.
- Moved standalone utilities (`install.py`, `finetune_peft.py`, `load_compressed_model.py`, `clean_dataset.py`) into a dedicated `scripts/` directory.
- Moved auxiliary documentation (`INSTALACION_RAPIDA.md`, `VERIFICAR_INSTALACION.md`, etc.) into `docs/` with English filenames.
- Consolidated all test files into the `tests/` directory.
- Removed build artifacts (`build/`, `dist/`, `.egg-info/`) from the repository.
- Replaced `black` formatter with `ruff format` for unified linting and formatting.
- Switched test runner from `pytest` to the standard library `unittest` framework.

### Fixed

- Recursion-safe model saving with `safetensors` fallback and component-based serialization.
- WSL2 GPU detection and CUDA availability checks in `install.py`.
- All critical lint errors: F821 (undefined names), E722 (bare except), E402 (import order), F401/F811 (unused/redundant imports), B019 (lru_cache on methods), and B904 (raise without from).
- Resolved broken `torchvision` installation causing `ModuleNotFoundError` for `transformers` imports.
- Refactored E2E Ollama server tests to use `fastapi.testclient.TestClient` instead of subprocess spawning.
- Fixed dataset-manager tests to avoid interactive `input()` dependency in CI.
- Skipped tensorly-dependent MPO tests when the optional dependency is not installed.
- Updated all internal import paths after moving scripts into `scripts/` and `tests/` directories.
