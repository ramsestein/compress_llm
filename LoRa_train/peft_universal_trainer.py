#!/usr/bin/env python3
"""
Entrenador universal para todos los métodos PEFT
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
import time
import gc

# Importar configuraciones y métodos PEFT
from .peft_methods_config import (
    PEFTMethod, BasePEFTConfig, LoRAConfig, MoLoRAConfig, 
    GaLoreConfig, DoRAConfig, AdaLoRAConfig, BitFitConfig,
    IA3Config, PromptTuningConfig, AdapterConfig, QLoRAConfig
)
from .peft_methods import (
    create_peft_model, MoLoRALinear, GaLoreLinear, DoRALinear,
    AdaLoRALinear, BitFitModel, IA3Linear, PromptEncoder, AdapterLayer
)

logger = logging.getLogger(__name__)


class PEFTProgressCallback(TrainerCallback):
    """Callback para mostrar progreso específico de PEFT"""
    
    def __init__(self, peft_method: PEFTMethod, total_steps: int):
        self.peft_method = peft_method
        self.total_steps = total_steps
        self.progress_bar = None
        self.current_loss = 0
        self.method_specific_metrics = {}
        
    def on_train_begin(self, args, state, control, **kwargs):
        desc = f"Entrenando {self.peft_method.value.upper()}"
        self.progress_bar = tqdm(total=self.total_steps, desc=desc)
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0:
            self.progress_bar.update(1)
            
            # Actualizar loss
            # Verificar que log_history existe y no está vacío
            if hasattr(state, 'log_history') and state.log_history:
                # Buscar el log más reciente con loss
                for i in range(len(state.log_history) - 1, -1, -1):
                    if isinstance(state.log_history[i], dict) and 'loss' in state.log_history[i]:
                        self.current_loss = state.log_history[i]['loss']
                        break
            
            # Métricas
            metrics = {}
            if self.current_loss > 0:
                metrics['loss'] = f'{self.current_loss:.4f}'
            
            # Métricas específicas por método
            # AdaLoRA: mostrar rango actual
            if self.peft_method == PEFTMethod.ADALORA and model:
                for name, module in model.named_modules():
                    if isinstance(module, AdaLoRALinear):
                        active_ranks = module.rank_mask.sum().item()
                        metrics['active_ranks'] = int(active_ranks)
                        break
            
            # MoLoRA: mostrar balance de expertos
            elif self.peft_method == PEFTMethod.MOLORA and model:
                # TODO: Implementar métricas de router
                pass
            
            if metrics:
                self.progress_bar.set_postfix(metrics)
        
    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar:
            self.progress_bar.close()


class UniversalPEFTTrainer:
    """Entrenador universal para cualquier método PEFT"""
    
    def __init__(self, config: BasePEFTConfig, model_path: Path, output_dir: Path):
        self.config = config
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Componentes
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Verificar compatibilidad
        self._check_compatibility()
    
    def _check_compatibility(self):
        """Verifica compatibilidad del método con el hardware"""
        if self.config.method == PEFTMethod.QLORA and not torch.cuda.is_available():
            raise RuntimeError("QLoRA requiere GPU con CUDA")
        
        if self.config.method == PEFTMethod.GALORE and self.device.type != "cuda":
            logger.warning("GaLore es más eficiente en GPU")
    
    def train(self, training_data: List[Dict[str, str]], 
              eval_data: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Entrena el modelo con el método PEFT especificado"""
        
        start_time = time.time()
        
        try:
            # 1. Cargar modelo y tokenizer
            logger.info(f"Cargando modelo para {self.config.method.value}...")
            self._load_model_and_tokenizer()
            
            # 2. Aplicar método PEFT
            logger.info(f"Aplicando {self.config.method.value}...")
            self._apply_peft_method()
            
            # 3. Preparar datos
            logger.info("Preparando datasets...")
            train_dataset = self._prepare_dataset(training_data)
            eval_dataset = self._prepare_dataset(eval_data) if eval_data else None
            
            # 4. Configurar entrenamiento
            logger.info("Configurando entrenamiento...")
            training_args = self._create_training_args(len(train_dataset))
            
            # 5. Crear trainer
            trainer = self._create_trainer(
                training_args,
                train_dataset,
                eval_dataset
            )
            
            # 6. Manejar entrenamiento específico por método
            if self.config.method == PEFTMethod.ADALORA:
                # AdaLoRA necesita actualizaciones durante el entrenamiento
                train_result = self._train_adalora(trainer)
            elif self.config.method == PEFTMethod.GALORE:
                # GaLore usa optimizador especial
                train_result = self._train_galore(trainer)
            else:
                # Entrenamiento estándar
                train_result = trainer.train()
            
            # 7. Guardar modelo
            logger.info("Guardando modelo...")
            output_path = self._save_model(trainer)
            
            # 8. Calcular métricas finales
            metrics = {
                'output_dir': str(output_path),
                'training_time': (time.time() - start_time) / 60,
                'final_loss': train_result.training_loss,
                'total_steps': train_result.global_step,
                'method': self.config.method.value,
                'trainable_params': self._count_trainable_params()
            }
            
            # Limpiar memoria
            self._cleanup()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            self._cleanup()
            raise
    
    def _load_model_and_tokenizer(self):
        """Carga modelo y tokenizer según el método"""
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración de carga según método
        if self.config.method == PEFTMethod.QLORA:
            # QLoRA necesita cuantización
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.double_quant,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=self.config.compute_dtype
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
        elif self.config.method == PEFTMethod.GALORE:
            # GaLore prefiere FP32 para proyecciones precisas
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
        else:
            # Carga estándar
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
    
    def _apply_peft_method(self):
        """Aplica el método PEFT al modelo"""
        if self.config.method in [PEFTMethod.LORA, PEFTMethod.QLORA]:
            # Usar PEFT library para LoRA estándar
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            if self.config.method == PEFTMethod.QLORA:
                self.model = prepare_model_for_kbit_training(self.model)
            
            peft_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                target_modules=self.config.target_modules
            )
            
            self.model = get_peft_model(self.model, peft_config)
            
        else:
            # Usar implementaciones personalizadas
            self.model = create_peft_model(self.model, self.config)
        
        # Imprimir información
        self._print_trainable_parameters()
    
    def _prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """Prepara dataset para entrenamiento"""
        if not data:
            return None
        
        # Para Prompt Tuning, agregar tokens virtuales
        if self.config.method == PEFTMethod.PROMPT_TUNING:
            # Los prompts se manejan en el modelo
            pass
        
        # Crear dataset
        dataset = Dataset.from_list(data)
        
        # Tokenizar
        def tokenize_function(examples):
            # Preparar textos para tokenización
            texts = []
            
            # Verificar si es un batch o un solo ejemplo
            is_batched = isinstance(examples.get('text', None), list) if 'text' in examples else False
            
            if not is_batched:
                # Convertir a formato batch
                examples = {k: [v] for k, v in examples.items()}
            
            # Determinar el número de ejemplos
            num_examples = len(next(iter(examples.values())))
            
            for i in range(num_examples):
                # Manejar diferentes formatos de entrada
                if 'text' in examples and examples['text'][i]:
                    texts.append(examples['text'][i])
                elif 'instruction' in examples and 'response' in examples:
                    instruction = examples['instruction'][i] if examples['instruction'][i] else ""
                    response = examples['response'][i] if examples['response'][i] else ""
                    
                    # Formato de chat
                    if 'system' in examples and examples['system'][i]:
                        text = f"{examples['system'][i]}\n\nHuman: {instruction}\n\nAssistant: {response}"
                    else:
                        text = f"Human: {instruction}\n\nAssistant: {response}"
                    texts.append(text)
                else:
                    # Si no hay formato reconocido, intentar concatenar todos los campos
                    text_parts = []
                    for key, values in examples.items():
                        if values[i]:
                            text_parts.append(str(values[i]))
                    if text_parts:
                        texts.append(" ".join(text_parts))
                    else:
                        texts.append("")  # Texto vacío si no hay nada
            
            # Si no hay textos válidos, usar un placeholder
            if not texts or all(not t for t in texts):
                texts = ["[EMPTY]"] * num_examples
            
            # Tokenizar
            return self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=getattr(self.config, 'max_seq_length', 512),
                return_tensors=None
            )
        
        # Aplicar tokenización
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Agregar labels para language modeling
        def add_labels(examples):
            examples['labels'] = examples['input_ids'].copy()
            return examples
        
        tokenized = tokenized.map(add_labels, batched=True)
        
        return tokenized
    
    def _create_training_args(self, dataset_size: int) -> TrainingArguments:
        """Crea argumentos de entrenamiento"""
        steps_per_epoch = dataset_size // self.config.per_device_train_batch_size
        total_steps = steps_per_epoch * self.config.num_train_epochs
        
        # Nombre único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.model_path.name}_{self.config.method.value}_{timestamp}"
        
        # Directorio de salida
        output_dir = self.output_dir / run_name
        
        # Configuración base
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if hasattr(self.config, 'do_eval') and self.config.do_eval else None,
            save_total_limit=3,
            fp16=False,
            #Comentar si no soportado bf16 por CUDA
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            gradient_checkpointing=self.config.gradient_checkpointing,
            remove_unused_columns=False,
            save_safetensors=False,
            report_to="none",
            run_name=run_name
        )
        
        # Ajustes específicos por método
        if self.config.method == PEFTMethod.QLORA:
            args.optim = self.config.optim
            args.gradient_checkpointing = True
        
        return args
    
    def _create_trainer(self, training_args: TrainingArguments,
                       train_dataset: Dataset,
                       eval_dataset: Optional[Dataset] = None) -> Trainer:
        """Crea trainer con configuración específica del método"""
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Callbacks
        callbacks = [
            PEFTProgressCallback(
                self.config.method,
                training_args.max_steps or 
                (len(train_dataset) // training_args.per_device_train_batch_size) * 
                training_args.num_train_epochs
            )
        ]
        
        # Optimizador personalizado para algunos métodos
        optimizers = (None, None)
        
        if self.config.method == PEFTMethod.GALORE:
            # GaLore necesita optimizador especial
            optimizers = self._create_galore_optimizer(training_args)
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
            optimizers=optimizers
        )
        
        return trainer
    
    def _train_adalora(self, trainer: Trainer) -> Any:
        """Entrenamiento especial para AdaLoRA con actualización de rangos"""
        
        # Hook para actualizar rangos
        def update_adalora_ranks(model, global_step):
            for name, module in model.named_modules():
                if isinstance(module, AdaLoRALinear):
                    module.update_rank_importance(global_step)
                    if global_step % self.config.deltaT == 0:
                        module.prune_ranks(global_step)
        
        # Modificar el step del trainer
        original_training_step = trainer.training_step
        
        def training_step_with_adalora(model, inputs):
            loss = original_training_step(model, inputs)
            update_adalora_ranks(model, trainer.state.global_step)
            return loss
        
        trainer.training_step = training_step_with_adalora
        
        # Entrenar
        return trainer.train()
    
    def _train_galore(self, trainer: Trainer) -> Any:
        """Entrenamiento especial para GaLore"""
        # El manejo especial ya está en GaLoreLinear con hooks
        return trainer.train()
    
    def _create_galore_optimizer(self, training_args: TrainingArguments):
        """Crea optimizador para GaLore"""
        # Parámetros que necesitan proyección
        galore_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(target in name for target in self.config.target_modules or []):
                    galore_params.append(param)
                else:
                    other_params.append(param)
        
        # Optimizador
        if self.config.optimizer == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit
            optimizer_class = AdamW8bit
        else:
            from torch.optim import AdamW
            optimizer_class = AdamW
        
        optimizer = optimizer_class([
            {'params': galore_params, 'lr': training_args.learning_rate},
            {'params': other_params, 'lr': training_args.learning_rate}
        ], weight_decay=training_args.weight_decay)
        
        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps
        )
        
        return optimizer, scheduler
    
    def _save_model(self, trainer: Trainer) -> Path:
        """Guarda el modelo según el método"""
        output_path = Path(trainer.args.output_dir)
        
        # Guardar modelo
        if self.config.method in [PEFTMethod.LORA, PEFTMethod.QLORA]:
            # LoRA guarda solo adaptadores
            trainer.save_model()
        else:
            # Otros métodos pueden necesitar guardado especial
            self.model.save_pretrained(output_path)
        
        # Guardar tokenizer
        self.tokenizer.save_pretrained(output_path)
        
        # Guardar configuración PEFT
        # Convertir el config a un diccionario serializable
        config_dict_raw = self.config.__dict__.copy()
        
        # Convertir tipos no serializables
        config_dict = {}
        for key, value in config_dict_raw.items():
            if isinstance(value, PEFTMethod):
                config_dict[key] = value.value  # Convertir enum a string
            elif isinstance(value, torch.dtype):
                config_dict[key] = str(value)  # Convertir dtype a string
            elif hasattr(value, '__dict__'):
                # Para objetos complejos, intentar convertir a dict
                try:
                    config_dict[key] = value.__dict__
                except:
                    config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        # Crear el diccionario final
        final_config = {
            'method': self.config.method.value,  # Asegurar que method es string
            'config': config_dict,
            'model_base': str(self.model_path),
            'training_completed': datetime.now().isoformat()
        }
        
        with open(output_path / 'peft_config.json', 'w') as f:
            json.dump(final_config, f, indent=2, default=str)  # default=str para manejar cualquier tipo no serializable
    
    def _count_trainable_params(self) -> int:
        """Cuenta parámetros entrenables"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _print_trainable_parameters(self):
        """Imprime información sobre parámetros entrenables"""
        trainable_params = self._count_trainable_params()
        all_param = sum(p.numel() for p in self.model.parameters())
        
        logger.info(
            f"Parámetros entrenables: {trainable_params:,} || "
            f"Parámetros totales: {all_param:,} || "
            f"Porcentaje: {100 * trainable_params / all_param:.2f}%"
        )
    
    def _cleanup(self):
        """Limpia memoria"""
        if self.model:
            del self.model
        if self.optimizer:
            del self.optimizer
        if self.scheduler:
            del self.scheduler
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Funciones de utilidad para cargar modelos entrenados

def load_peft_model(model_path: str, device: Optional[str] = None):
    """Carga un modelo entrenado con cualquier método PEFT"""
    model_path = Path(model_path)
    
    # Leer configuración
    with open(model_path / 'peft_config.json', 'r') as f:
        config = json.load(f)
    
    method = PEFTMethod(config['method'])
    
    # Cargar según método
    if method in [PEFTMethod.LORA, PEFTMethod.QLORA]:
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model_base'],
            device_map=device or "auto"
        )
        
        model = PeftModel.from_pretrained(base_model, str(model_path))
        
    else:
        # Cargar modelo completo
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device or "auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    return model, tokenizer, config