"""
Módulo LoRA y PEFT para fine-tuning eficiente de modelos de lenguaje
"""

# Importar clases principales de configuración
from .lora_config import (
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    LoRAPresets,
    get_model_specific_config,
    TaskType
)

# Importar configuraciones PEFT
from .peft_methods_config import (
    PEFTMethod,
    BasePEFTConfig,
    LoRAConfig as PEFTLoRAConfig,
    MoLoRAConfig,
    GaLoreConfig,
    DoRAConfig,
    BitFitConfig,
    IA3Config,
    PromptTuningConfig,
    AdapterConfig,
    QLoRAConfig,
    CompacterConfig,
    KronAConfig,
    S4Config,
    HoulsbyConfig,
    PEFTPresets,
    get_config_by_name
)

# Importar métodos PEFT
from .peft_methods import (
    BasePEFTModule,
    MoLoRARouter,
    MoLoRALinear,
    GaLoreProjector,
    GaLoreLinear,
    DoRALinear,
    BitFitModel,
    IA3Linear,
    PromptEncoder,
    AdapterLinear,
    QuantizedLoRALinear,
    PrunedLoRALinear,
    CompacterLinear,
    KronALinear,
    S4Adapter,
    HoulsbyAdapterLinear,
    create_peft_model
)

# Importar entrenadores
from .lora_trainer import LoRATrainer
from .peft_universal_trainer import PEFTUniversalTrainer

# Importar gestor de datasets
from .dataset_manager import (
    DatasetConfig,
    OptimizedDatasetManager
)

__version__ = "0.1.0"

__all__ = [
    # Configuraciones LoRA
    "LoRAConfig",
    "TrainingConfig", 
    "DataConfig",
    "LoRAPresets",
    "get_model_specific_config",
    "TaskType",
    
    # Configuraciones PEFT
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
    
    # Métodos PEFT
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
    
    # Entrenadores
    "LoRATrainer",
    "PEFTUniversalTrainer",
    
    # Dataset manager
    "DatasetConfig",
    "OptimizedDatasetManager"
]