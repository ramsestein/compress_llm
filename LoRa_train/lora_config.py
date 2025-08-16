"""
Configuración y parámetros para entrenamiento LoRA
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class TaskType(Enum):
    """Tipos de tareas soportadas"""
    CAUSAL_LM = "causal_lm"  # Generación de texto
    SEQ_CLS = "seq_cls"      # Clasificación
    TOKEN_CLS = "token_cls"  # NER
    QA = "qa"                # Question Answering

@dataclass
class LoRAConfig:
    """Configuración para LoRA (Low-Rank Adaptation)"""
    # Parámetros básicos de LoRA
    r: int = 16                          # Rango de las matrices
    lora_alpha: int = 32                 # Parámetro de escalado
    lora_dropout: float = 0.1            # Dropout para LoRA
    bias: str = "none"                   # none, all, lora_only
    task_type: str = "CAUSAL_LM"         # Tipo de tarea
    
    # Módulos objetivo (qué capas adaptar)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Atención
        "gate_proj", "up_proj", "down_proj",     # FFN
        "lm_head"                                 # Output
    ])
    
    # Módulos a excluir
    modules_to_save: List[str] = field(default_factory=lambda: [
        "embed_tokens",
        "lm_head"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'bias': self.bias,
            'task_type': self.task_type,
            'target_modules': self.target_modules,
            'modules_to_save': self.modules_to_save
        }

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    # Básicos
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Optimización
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    
    # Logging y guardado
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Optimizaciones de memoria
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False  # Usar si la GPU soporta
    optim: str = "paged_adamw_8bit"  # Optimizador eficiente
    
    # Otros
    max_seq_length: int = 512
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    label_smoothing_factor: float = 0.0
    
    # Evaluación
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    greater_is_better: bool = False
    metric_for_best_model: str = "eval_loss"
    load_best_model_at_end: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

@dataclass
class DataConfig:
    """Configuración de datos"""
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Templates de formato
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    chat_template: str = "<|user|>\n{instruction}\n<|assistant|>\n{response}"
    
    # Proporción train/eval
    eval_split_ratio: float = 0.1
    shuffle: bool = True
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

class LoRAPresets:
    """Presets de configuración para diferentes casos de uso"""
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """Obtiene un preset de configuración"""
        presets = {
            "conservative": {
                "lora": LoRAConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj"]
                ),
                "training": TrainingConfig(
                    num_train_epochs=1,
                    learning_rate=1e-4,
                    per_device_train_batch_size=2
                ),
                "data": DataConfig()
            },
            "balanced": {
                "lora": LoRAConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1
                ),
                "training": TrainingConfig(
                    num_train_epochs=3,
                    learning_rate=2e-4,
                    per_device_train_batch_size=4
                ),
                "data": DataConfig()
            },
            "aggressive": {
                "lora": LoRAConfig(
                    r=32,
                    lora_alpha=64,
                    lora_dropout=0.1,
                    target_modules=[
                        "q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"
                    ]
                ),
                "training": TrainingConfig(
                    num_train_epochs=5,
                    learning_rate=3e-4,
                    per_device_train_batch_size=8,
                    gradient_accumulation_steps=2
                ),
                "data": DataConfig(
                    max_length=1024
                )
            },
            "healing": {
                "lora": LoRAConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=[
                        "transformer.h.0.attn.c_attn",
                        "transformer.h.0.attn.c_proj",
                        "transformer.h.0.mlp.c_fc",
                        "transformer.h.0.mlp.c_proj"
                    ]
                ),
                "training": TrainingConfig(
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    per_device_train_batch_size=2,
                    save_steps=100
                ),
                "data": DataConfig(
                    max_length=256
                )
            }
        }
        
        return presets.get(preset_name, presets["balanced"])
    
    @staticmethod
    def estimate_memory_usage(model_size_gb: float, config: LoRAConfig, 
                            batch_size: int = 4) -> Dict[str, float]:
        """Estima el uso de memoria para el entrenamiento"""
        # Estimación aproximada
        base_model_memory = model_size_gb  # Modelo base
        
        # LoRA añade ~0.1-1% de parámetros según r
        lora_params_ratio = (config.r / 1024) * len(config.target_modules) * 0.01
        lora_memory = model_size_gb * lora_params_ratio
        
        # Memoria para gradientes y optimizador
        gradient_memory = (base_model_memory + lora_memory) * 2  # Aproximado
        optimizer_memory = lora_memory * 4  # Estados del optimizador
        
        # Activaciones (depende del batch size)
        activation_memory = batch_size * 0.5  # ~0.5GB por muestra
        
        total_memory = (
            base_model_memory +  # Modelo
            lora_memory +        # Adaptadores LoRA
            gradient_memory +    # Gradientes
            optimizer_memory +   # Optimizador
            activation_memory    # Activaciones
        )
        
        return {
            'base_model': base_model_memory,
            'lora_adapters': lora_memory,
            'gradients': gradient_memory,
            'optimizer': optimizer_memory,
            'activations': activation_memory,
            'total_estimated': total_memory,
            'recommended_gpu_memory': total_memory * 1.2  # 20% margen
        }

def get_model_specific_config(model_name: str) -> Dict[str, List[str]]:
    """Obtiene configuración específica según el tipo de modelo"""
    model_configs = {
        "llama": {
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"]
        },
        "mistral": {
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"]
        },
        "gpt2": {
            "target_modules": ["c_attn", "c_proj", "c_fc"],
            "modules_to_save": ["wte", "wpe", "ln_f"]
        },
        "bert": {
            "target_modules": ["query", "key", "value", "dense"],
            "modules_to_save": ["embeddings", "pooler"]
        },
        "t5": {
            "target_modules": ["q", "v", "k", "o", "wi", "wo"],
            "modules_to_save": ["shared", "lm_head"]
        },
        "mbart": {  # Para NLLB
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            "modules_to_save": ["embed_tokens", "embed_positions", "lm_head"]
        }
    }
    
    # Detectar tipo de modelo
    model_lower = model_name.lower()
    for model_type, config in model_configs.items():
        if model_type in model_lower:
            return config
    
    # Default para modelos desconocidos
    return model_configs["llama"]