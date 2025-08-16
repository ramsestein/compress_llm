#!/usr/bin/env python3
"""
Configuraciones para métodos PEFT universales
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import torch


class PEFTMethod(Enum):
    """Métodos PEFT disponibles"""
    LORA = "lora"
    MOLORA = "molora"
    GALORE = "galore"
    DORA = "dora"
    ADALORA = "adalora"
    BITFIT = "bitfit"
    IA3 = "ia3"
    PROMPT_TUNING = "prompt_tuning"
    ADAPTER = "adapter"
    QLORA = "qlora"


@dataclass
class BasePEFTConfig:
    """Configuración base para todos los métodos PEFT"""
    method: PEFTMethod
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    seed: int = 42


@dataclass
class LoRAConfig(BasePEFTConfig):
    """Configuración para LoRA"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.LORA


@dataclass
class MoLoRAConfig(BasePEFTConfig):
    """Configuración para MoLoRA (Mixture of LoRAs)"""
    num_experts: int = 4
    expert_r: List[int] = None
    expert_alpha: List[int] = None
    expert_dropout: List[float] = None
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.MOLORA
        if self.expert_r is None:
            self.expert_r = [8] * self.num_experts
        if self.expert_alpha is None:
            self.expert_alpha = [16] * self.num_experts
        if self.expert_dropout is None:
            self.expert_dropout = [0.1] * self.num_experts


@dataclass
class GaLoreConfig(BasePEFTConfig):
    """Configuración para GaLore (Gradient Low-Rank)"""
    rank: int = 128
    scale: float = 0.25
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.GALORE


@dataclass
class DoRAConfig(BasePEFTConfig):
    """Configuración para DoRA (Decomposed LoRA)"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    magnitude_lr_scale: float = 0.1
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.DORA


@dataclass
class AdaLoRAConfig(BasePEFTConfig):
    """Configuración para AdaLoRA (Adaptive LoRA)"""
    init_r: int = 64
    target_r: int = 16
    tinit: int = 200
    tfinal: int = 1000
    deltaT: int = 10
    beta1: float = 0.85
    beta2: float = 0.85
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.ADALORA


@dataclass
class BitFitConfig(BasePEFTConfig):
    """Configuración para BitFit (Bias Tuning)"""
    train_embeddings: bool = False
    train_layer_norms: bool = True
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.BITFIT


@dataclass
class IA3Config(BasePEFTConfig):
    """Configuración para IA³ (Infused Adapter)"""
    init_ia3_weights: str = "ones"
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.IA3


@dataclass
class PromptTuningConfig(BasePEFTConfig):
    """Configuración para Prompt Tuning"""
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "random"
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.PROMPT_TUNING


@dataclass
class AdapterConfig(BasePEFTConfig):
    """Configuración para Adapter Tuning"""
    adapter_size: int = 64
    adapter_type: str = "pfeiffer"
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.ADAPTER


@dataclass
class QLoRAConfig(BasePEFTConfig):
    """Configuración para QLoRA (Quantized LoRA)"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bits: int = 4
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        self.method = PEFTMethod.QLORA


# Presets predefinidos
PEFTPresets = {
    "efficient": {
        "lora": LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=1e-4,
            num_train_epochs=2
        ),
        "bitfit": BitFitConfig(
            method=PEFTMethod.BITFIT,
            learning_rate=5e-5,
            num_train_epochs=1
        ),
        "ia3": IA3Config(
            method=PEFTMethod.IA3,
            learning_rate=1e-3,
            num_train_epochs=1
        )
    },
    
    "balanced": {
        "lora": LoRAConfig(
            method=PEFTMethod.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            learning_rate=2e-4,
            num_train_epochs=3
        ),
        "dora": DoRAConfig(
            method=PEFTMethod.DORA,
            r=16,
            lora_alpha=32,
            learning_rate=2e-4,
            num_train_epochs=3
        ),
        "adapter": AdapterConfig(
            method=PEFTMethod.ADAPTER,
            adapter_size=64,
            learning_rate=1e-4,
            num_train_epochs=3
        )
    },
    
    "quality": {
        "lora": LoRAConfig(
            method=PEFTMethod.LORA,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            learning_rate=1e-4,
            num_train_epochs=5
        ),
        "adalora": AdaLoRAConfig(
            method=PEFTMethod.ADALORA,
            init_r=128,
            target_r=32,
            learning_rate=1e-4,
            num_train_epochs=5
        ),
        "molora": MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=8,
            expert_r=[16, 16, 32, 32, 64, 64, 128, 128],
            learning_rate=1e-4,
            num_train_epochs=5
        )
    },
    
    "memory_efficient": {
        "qlora": QLoRAConfig(
            method=PEFTMethod.QLORA,
            r=8,
            lora_alpha=16,
            bits=4,
            learning_rate=1e-4,
            num_train_epochs=2
        ),
        "galore": GaLoreConfig(
            method=PEFTMethod.GALORE,
            rank=64,
            scale=0.25,
            learning_rate=5e-5,
            num_train_epochs=1
        ),
        "prompt_tuning": PromptTuningConfig(
            method=PEFTMethod.PROMPT_TUNING,
            num_virtual_tokens=10,
            learning_rate=1e-3,
            num_train_epochs=1
        )
    }
}


def get_config_by_name(preset_name: str, method_name: str) -> BasePEFTConfig:
    """Obtiene configuración por nombre de preset y método"""
    if preset_name not in PEFTPresets:
        raise ValueError(f"Preset '{preset_name}' no encontrado")
    
    if method_name not in PEFTPresets[preset_name]:
        raise ValueError(f"Método '{method_name}' no encontrado en preset '{preset_name}'")
    
    return PEFTPresets[preset_name][method_name]


def get_available_methods(preset_name: str) -> List[str]:
    """Obtiene métodos disponibles para un preset"""
    if preset_name not in PEFTPresets:
        return []
    
    return list(PEFTPresets[preset_name].keys())


def get_available_presets() -> List[str]:
    """Obtiene presets disponibles"""
    return list(PEFTPresets.keys())


def estimate_memory_usage(config: BasePEFTConfig, model_size_billions: float) -> Dict[str, float]:
    """Estima uso de memoria para una configuración"""
    model_size_gb = model_size_billions * 2  # FP16
    
    if isinstance(config, LoRAConfig):
        # LoRA: r * embedding_dim * num_modules * 2 (A + B) * 2 bytes
        estimated_params = config.r * 4096 * 7 * 2 * 2  # Aproximado
        memory_mb = (estimated_params * 4) / (1024 * 1024)  # FP32
    elif isinstance(config, BitFitConfig):
        # BitFit: ~0.1% del modelo
        memory_mb = model_size_gb * 1024 * 0.001
    elif isinstance(config, PromptTuningConfig):
        # Prompt Tuning: tokens * embedding_dim
        memory_mb = (config.num_virtual_tokens * 4096 * 4) / (1024 * 1024)
    elif isinstance(config, QLoRAConfig):
        # QLoRA: LoRA + 4-bit quantization
        memory_mb = model_size_gb * 1024 * 0.25  # 4-bit
    else:
        # Otros métodos: estimación conservadora
        memory_mb = model_size_gb * 1024 * 0.1
    
    return {
        "model_memory_gb": model_size_gb,
        "peft_memory_mb": memory_mb,
        "total_memory_gb": model_size_gb + (memory_mb / 1024)
    }