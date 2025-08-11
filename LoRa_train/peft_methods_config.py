#!/usr/bin/env python3
"""
Configuración para múltiples métodos PEFT (Parameter-Efficient Fine-Tuning)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import torch

class PEFTMethod(Enum):
    """Métodos de fine-tuning eficiente disponibles"""
    LORA = "lora"
    MOLORA = "molora"          # Mixture of LoRAs
    GALORE = "galore"          # Gradient Low-Rank Projection
    DORA = "dora"              # Weight-Decomposed LoRA
    ADALORA = "adalora"        # Adaptive LoRA
    BITFIT = "bitfit"          # Solo entrenar bias
    IA3 = "ia3"                # Infused Adapter by Inhibiting and Amplifying
    PROMPT_TUNING = "prompt_tuning"
    ADAPTER = "adapter"        # Adapter tuning
    QLORA = "qlora"           # Quantized LoRA

@dataclass
class BasePEFTConfig:
    """Configuración base para todos los métodos PEFT"""
    method: PEFTMethod
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Módulos objetivo comunes
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None

@dataclass
class LoRAConfig(BasePEFTConfig):
    """Configuración estándar de LoRA"""
    method: PEFTMethod = PEFTMethod.LORA
    r: int = 16                          # Rango
    lora_alpha: int = 32                 # Scaling
    lora_dropout: float = 0.1
    bias: str = "none"                   # none, all, lora_only
    fan_in_fan_out: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]

@dataclass 
class MoLoRAConfig(LoRAConfig):
    """Configuración para Mixture of LoRAs"""
    method: PEFTMethod = PEFTMethod.MOLORA
    num_experts: int = 4                 # Número de expertos LoRA
    router_type: str = "learned"         # learned, hash, random
    load_balancing_loss_weight: float = 0.01
    expert_capacity_factor: float = 1.25
    router_z_loss_weight: float = 0.001  # Para regularización del router
    
    # Configuración por experto
    expert_r: List[int] = None           # Rango por experto
    expert_alpha: List[int] = None       # Alpha por experto
    
    def __post_init__(self):
        super().__post_init__()
        if self.expert_r is None:
            self.expert_r = [self.r] * self.num_experts
        if self.expert_alpha is None:
            self.expert_alpha = [self.lora_alpha] * self.num_experts

@dataclass
class GaLoreConfig(BasePEFTConfig):
    """Configuración para Gradient Low-Rank Projection"""
    method: PEFTMethod = PEFTMethod.GALORE
    rank: int = 128                      # Rango de proyección
    update_proj_gap: int = 200           # Actualizar proyección cada N pasos
    scale: float = 0.25                  # Factor de escala
    proj_type: str = "std"               # std, reverse_std, right, left, full
    
    # Optimizador especial para GaLore
    optimizer: str = "adamw8bit"         # adamw, adamw8bit, adafactor
    scheduler: str = "linear"            # linear, cosine, constant
    
    def __post_init__(self):
        # GaLore puede aplicarse a todas las matrices de pesos
        if self.target_modules is None:
            self.target_modules = ["attn", "mlp", "layernorm", "embedding"]

@dataclass
class DoRAConfig(LoRAConfig):
    """Configuración para Weight-Decomposed LoRA"""
    method: PEFTMethod = PEFTMethod.DORA
    # DoRA hereda la mayoría de parámetros de LoRA
    # pero descompone W = magnitude * direction
    magnitude_lr_scale: float = 0.1      # LR relativo para magnitud
    direction_lr_scale: float = 1.0      # LR relativo para dirección
    orthogonal_init: bool = True         # Inicialización ortogonal
    normalize_direction: bool = True     # Normalizar vectores de dirección

@dataclass
class AdaLoRAConfig(LoRAConfig):
    """Configuración para Adaptive LoRA"""
    method: PEFTMethod = PEFTMethod.ADALORA
    init_r: int = 64                     # Rango inicial (se reduce adaptivamente)
    target_r: int = 16                   # Rango objetivo final
    tinit: int = 200                     # Pasos de calentamiento para adaptación
    tfinal: int = 1000                   # Pasos finales de adaptación
    deltaT: int = 10                     # Intervalo de actualización
    beta1: float = 0.85                  # Para importancia de módulos
    beta2: float = 0.85                  # Para importancia de rangos
    orth_reg_weight: float = 0.5         # Regularización ortogonal
    
    # Método de poda de rangos
    prune_method: str = "magnitude"      # magnitude, random, importance
    mask_threshold: float = 0.1          # Umbral para máscara de poda

@dataclass
class BitFitConfig(BasePEFTConfig):
    """Configuración para BitFit (solo entrenar bias)"""
    method: PEFTMethod = PEFTMethod.BITFIT
    # BitFit es extremadamente simple - solo entrena bias
    bias_learning_rate_scale: float = 10.0  # Escalar LR para bias
    train_embeddings: bool = False           # También entrenar embeddings
    train_layer_norms: bool = True           # Entrenar layer norms
    
    def __post_init__(self):
        # BitFit no usa target_modules de la misma manera
        self.target_modules = None

@dataclass
class IA3Config(BasePEFTConfig):
    """Configuración para IA³ (Infused Adapter by Inhibiting and Amplifying)"""
    method: PEFTMethod = PEFTMethod.IA3
    init_ia3_weights: str = "ones"       # ones, zeros, normal, uniform
    feedforward_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # IA³ se aplica a key, value y FFN
            self.target_modules = ["k_proj", "v_proj"]
        if self.feedforward_modules is None:
            self.feedforward_modules = ["down_proj"]

@dataclass
class PromptTuningConfig(BasePEFTConfig):
    """Configuración para Prompt Tuning"""
    method: PEFTMethod = PEFTMethod.PROMPT_TUNING
    num_virtual_tokens: int = 20         # Número de tokens virtuales
    prompt_tuning_init: str = "random"   # random, text
    prompt_tuning_init_text: Optional[str] = None  # Si init="text"
    tokenizer_name_or_path: Optional[str] = None
    
    # Configuración del encoder de prompts
    encoder_hidden_size: Optional[int] = None  # None = usar embedding dim
    encoder_num_layers: int = 2
    encoder_reparameterization_type: str = "MLP"  # MLP, LSTM
    encoder_dropout: float = 0.0

@dataclass
class AdapterConfig(BasePEFTConfig):
    """Configuración para Adapter Tuning"""
    method: PEFTMethod = PEFTMethod.ADAPTER
    adapter_size: int = 64               # Tamaño del cuello de botella
    adapter_layers: List[str] = None     # Dónde insertar adapters
    adapter_type: str = "pfeiffer"      # pfeiffer, houlsby
    non_linearity: str = "relu"          # relu, gelu, swish
    adapter_dropout: float = 0.1
    
    # Configuración de inicialización
    init_weights: str = "bert"           # bert, xavier, normal
    scaling: float = 1.0                 # Factor de escala para residual
    
    def __post_init__(self):
        if self.adapter_layers is None:
            # Por defecto, agregar después de cada capa de atención y FFN
            self.adapter_layers = ["attention", "ffn"]

@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuración para Quantized LoRA"""
    method: PEFTMethod = PEFTMethod.QLORA
    # Hereda todo de LoRA más configuración de cuantización
    bits: int = 4                        # 4 u 8 bits
    double_quant: bool = True            # Doble cuantización
    quant_type: str = "nf4"              # fp4 o nf4
    compute_dtype: torch.dtype = torch.float16
    
    # Configuración específica de cuantización
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # Memoria y optimización
    optim: str = "paged_adamw_32bit"    # Optimizador paginado
    gradient_checkpointing: bool = True
    max_memory_MB: Optional[int] = None  # Límite de memoria

class PEFTPresets:
    """Presets optimizados para diferentes casos de uso"""
    
    @staticmethod
    def get_preset(method: PEFTMethod, use_case: str = "general") -> BasePEFTConfig:
        """Obtiene configuración preset según método y caso de uso"""
        
        presets = {
            PEFTMethod.LORA: {
                "general": LoRAConfig(r=16, lora_alpha=32),
                "efficient": LoRAConfig(r=8, lora_alpha=16),
                "quality": LoRAConfig(r=32, lora_alpha=64)
            },
            PEFTMethod.MOLORA: {
                "general": MoLoRAConfig(num_experts=4, r=16),
                "multitask": MoLoRAConfig(num_experts=8, r=8),
                "efficient": MoLoRAConfig(num_experts=2, r=16)
            },
            PEFTMethod.GALORE: {
                "general": GaLoreConfig(rank=128),
                "memory_constrained": GaLoreConfig(rank=64, update_proj_gap=500),
                "quality": GaLoreConfig(rank=256, update_proj_gap=100)
            },
            PEFTMethod.DORA: {
                "general": DoRAConfig(r=16, lora_alpha=32),
                "quality": DoRAConfig(r=32, lora_alpha=64, magnitude_lr_scale=0.05)
            },
            PEFTMethod.ADALORA: {
                "general": AdaLoRAConfig(init_r=64, target_r=16),
                "aggressive": AdaLoRAConfig(init_r=128, target_r=8),
                "conservative": AdaLoRAConfig(init_r=32, target_r=16)
            },
            PEFTMethod.BITFIT: {
                "general": BitFitConfig(),
                "minimal": BitFitConfig(train_layer_norms=False),
                "extended": BitFitConfig(train_embeddings=True)
            },
            PEFTMethod.IA3: {
                "general": IA3Config(),
                "minimal": IA3Config(feedforward_modules=[])
            },
            PEFTMethod.PROMPT_TUNING: {
                "general": PromptTuningConfig(num_virtual_tokens=20),
                "short": PromptTuningConfig(num_virtual_tokens=8),
                "long": PromptTuningConfig(num_virtual_tokens=50)
            },
            PEFTMethod.ADAPTER: {
                "general": AdapterConfig(adapter_size=64),
                "efficient": AdapterConfig(adapter_size=32),
                "quality": AdapterConfig(adapter_size=128)
            },
            PEFTMethod.QLORA: {
                "general": QLoRAConfig(r=16, lora_alpha=32, bits=4),
                "quality": QLoRAConfig(r=32, lora_alpha=64, bits=8),
                "extreme": QLoRAConfig(r=8, lora_alpha=16, bits=4)
            }
        }
        
        return presets.get(method, {}).get(use_case, presets[method]["general"])
    
    @staticmethod
    def estimate_memory_usage(config: BasePEFTConfig, model_size_gb: float) -> Dict[str, float]:
        """Estima uso de memoria según método y configuración"""
        
        # Factores de memoria por método
        memory_factors = {
            PEFTMethod.LORA: 0.1,
            PEFTMethod.MOLORA: 0.15,
            PEFTMethod.GALORE: 0.05,
            PEFTMethod.DORA: 0.12,
            PEFTMethod.ADALORA: 0.15,
            PEFTMethod.BITFIT: 0.01,
            PEFTMethod.IA3: 0.005,
            PEFTMethod.PROMPT_TUNING: 0.001,
            PEFTMethod.ADAPTER: 0.08,
            PEFTMethod.QLORA: 0.03
        }
        
        base_factor = memory_factors.get(config.method, 0.1)
        
        # Ajustar por configuración específica
        if hasattr(config, 'r'):
            base_factor *= (config.r / 16)  # Normalizado a r=16
        elif hasattr(config, 'adapter_size'):
            base_factor *= (config.adapter_size / 64)
        elif hasattr(config, 'num_virtual_tokens'):
            base_factor *= (config.num_virtual_tokens / 20)
        
        trainable_params_gb = model_size_gb * base_factor
        
        # Memoria total estimada
        memory_usage = {
            'model_base': model_size_gb,
            'trainable_params': trainable_params_gb,
            'gradients': trainable_params_gb,
            'optimizer_states': trainable_params_gb * 2,  # Adam tiene 2 estados
            'activations': model_size_gb * 0.1 * config.per_device_train_batch_size,
            'total_estimated': 0
        }
        
        # Ajustes especiales por método
        if config.method == PEFTMethod.QLORA:
            memory_usage['model_base'] *= 0.25  # 4-bit quantization
        elif config.method == PEFTMethod.GALORE:
            memory_usage['optimizer_states'] *= 0.1  # Proyección reduce estados
        
        memory_usage['total_estimated'] = sum(memory_usage.values())
        
        return memory_usage

def get_config_by_name(method_name: str, **kwargs) -> BasePEFTConfig:
    """Crea configuración por nombre del método"""
    method = PEFTMethod(method_name.lower())
    
    config_classes = {
        PEFTMethod.LORA: LoRAConfig,
        PEFTMethod.MOLORA: MoLoRAConfig,
        PEFTMethod.GALORE: GaLoreConfig,
        PEFTMethod.DORA: DoRAConfig,
        PEFTMethod.ADALORA: AdaLoRAConfig,
        PEFTMethod.BITFIT: BitFitConfig,
        PEFTMethod.IA3: IA3Config,
        PEFTMethod.PROMPT_TUNING: PromptTuningConfig,
        PEFTMethod.ADAPTER: AdapterConfig,
        PEFTMethod.QLORA: QLoRAConfig
    }
    
    config_class = config_classes.get(method)
    if config_class:
        return config_class(**kwargs)
    else:
        raise ValueError(f"Método no soportado: {method_name}")