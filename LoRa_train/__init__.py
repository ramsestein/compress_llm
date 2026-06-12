"""LoRA and PEFT module for efficient language model fine-tuning."""

# Import main configuration classes
# Import dataset manager
from .dataset_manager import DatasetConfig, OptimizedDatasetManager
from .lora_config import (
    DataConfig,
    LoRAConfig,
    LoRAPresets,
    TaskType,
    TrainingConfig,
    get_model_specific_config,
)

# Import trainers
from .lora_trainer import LoRATrainer

# Import PEFT methods
from .peft_methods import (
    AdapterLinear,
    BasePEFTModule,
    BitFitModel,
    CompacterLinear,
    DoRALinear,
    GaLoreLinear,
    GaLoreProjector,
    HoulsbyAdapterLinear,
    IA3Linear,
    KronALinear,
    MoLoRALinear,
    MoLoRARouter,
    PromptEncoder,
    PrunedLoRALinear,
    QuantizedLoRALinear,
    S4Adapter,
    create_peft_model,
)

# Import PEFT configurations
from .peft_methods_config import (
    AdapterConfig,
    BasePEFTConfig,
    BitFitConfig,
    CompacterConfig,
    DoRAConfig,
    GaLoreConfig,
    HoulsbyConfig,
    IA3Config,
    KronAConfig,
    MoLoRAConfig,
    PEFTMethod,
    PEFTPresets,
    PromptTuningConfig,
    QLoRAConfig,
    S4Config,
    get_config_by_name,
)
from .peft_universal_trainer import PEFTUniversalTrainer

__version__ = "0.1.0"

__all__ = [
    # LoRA configurations
    "LoRAConfig",
    "TrainingConfig",
    "DataConfig",
    "LoRAPresets",
    "get_model_specific_config",
    "TaskType",
    # PEFT configurations
    "PEFTMethod",
    "BasePEFTConfig",
    "MoLoRAConfig",
    "GaLoreConfig",
    "DoRAConfig",
    "BitFitConfig",
    "IA3Config",
    "PromptTuningConfig",
    "AdapterConfig",
    "QLoRAConfig",
    "CompacterConfig",
    "KronAConfig",
    "S4Config",
    "HoulsbyConfig",
    "PEFTPresets",
    "get_config_by_name",
    # PEFT methods
    "BasePEFTModule",
    "MoLoRARouter",
    "MoLoRALinear",
    "GaLoreProjector",
    "GaLoreLinear",
    "DoRALinear",
    "BitFitModel",
    "IA3Linear",
    "PromptEncoder",
    "AdapterLinear",
    "QuantizedLoRALinear",
    "PrunedLoRALinear",
    "CompacterLinear",
    "KronALinear",
    "S4Adapter",
    "HoulsbyAdapterLinear",
    "create_peft_model",
    # Trainers
    "LoRATrainer",
    "PEFTUniversalTrainer",
    # Dataset manager
    "DatasetConfig",
    "OptimizedDatasetManager",
]
