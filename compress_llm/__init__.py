"""Compress LLM - A comprehensive suite for LLM compression and parameter-efficient fine-tuning.

This package provides tools for:
- Quantization, pruning, and tensor decomposition
- PEFT methods (LoRA, QLoRA, DoRA, MoLoRA, BitFit, IA3, etc.)
- Ollama-compatible local model serving
- Zero-code CLI for non-technical users
"""

__version__ = "1.0.0"
__author__ = "Ramses Marrero"
__license__ = "MIT"

from compress_llm.distillation import KnowledgeDistiller
from create_compress.compression_config_manager import CompressionConfigManager
from create_compress.compression_engine import CompressionEngine
from LoRa_train.dataset_manager import OptimizedDatasetManager
from LoRa_train.lora_trainer import LoRATrainer
from LoRa_train.peft_methods_config import (
    BitFitConfig,
    DoRAConfig,
    IA3Config,
    LoRAConfig,
    MoLoRAConfig,
    PEFTMethod,
    QLoRAConfig,
)

__all__ = [
    "__version__",
    "CompressionEngine",
    "CompressionConfigManager",
    "PEFTMethod",
    "LoRAConfig",
    "QLoRAConfig",
    "DoRAConfig",
    "MoLoRAConfig",
    "BitFitConfig",
    "IA3Config",
    "LoRATrainer",
    "OptimizedDatasetManager",
    "KnowledgeDistiller",
]
