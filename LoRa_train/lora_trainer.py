"""
Entrenador LoRA para fine-tuning de modelos
"""
import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
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
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import time
from torch.utils.data import DataLoader
import gc

logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    """Callback para mostrar progreso durante el entrenamiento"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.progress_bar = None
        self.current_loss = 0
        self.step_count = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=self.total_steps, desc="Entrenando")
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            
            # Actualizar métricas mostradas
            postfix = {}
            
            # Buscar loss en el state actual
            if hasattr(state, 'loss') and state.loss is not None:
                self.current_loss = state.loss
                postfix['loss'] = f'{self.current_loss:.4f}'
            
            # También buscar en log_history si existe
            elif state.log_history:
                # Buscar el log más reciente con loss
                for i in range(len(state.log_history) - 1, -1, -1):
                    if 'loss' in state.log_history[i]:
                        self.current_loss = state.log_history[i]['loss']
                        postfix['loss'] = f'{self.current_loss:.4f}'
                        break
            
            # Agregar información adicional
            postfix['step'] = f'{self.step_count}/{self.total_steps}'
            
            if postfix:
                self.progress_bar.set_postfix(postfix)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar:
            self.progress_bar.close()
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Callback cuando se registran logs"""
        if logs is not None and 'loss' in logs and self.progress_bar is not None:
            self.current_loss = logs['loss']
            self.progress_bar.set_postfix({'loss': f'{self.current_loss:.4f}'})

class LoRATrainer:
    """Entrenador para fine-tuning con LoRA"""
    
    def __init__(self, model_name: str, model_path: Path, output_dir: Path):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Componentes del modelo
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def train(self, training_data: List[Dict[str, str]], 
              lora_config: 'LoRAConfig',
              training_config: 'TrainingConfig',
              data_config: 'DataConfig') -> Dict[str, Any]:
        """Ejecuta el entrenamiento completo"""
        
        start_time = time.time()
        
        try:
            # 1. Cargar modelo y tokenizer
            logger.info("Cargando modelo base...")
            self._load_model_and_tokenizer()
            
            # 2. Preparar datos
            logger.info("Preparando dataset...")
            train_dataset, eval_dataset = self._prepare_datasets(
                training_data, 
                data_config
            )
            
            # 3. Configurar LoRA
            logger.info("Configurando LoRA...")
            self._setup_lora(lora_config)
            
            # 4. Configurar entrenamiento
            logger.info("Configurando entrenamiento...")
            training_args = self._create_training_args(
                training_config,
                len(train_dataset)
            )
            
            # 5. Crear trainer
            logger.info("Creando trainer...")
            trainer = self._create_trainer(
                training_args,
                train_dataset,
                eval_dataset,
                data_config
            )
            
            # 6. Entrenar
            logger.info("Iniciando entrenamiento...")
            train_result = trainer.train()
            
            # 7. Guardar modelo
            logger.info("Guardando modelo...")
            output_path = self._save_model(trainer, training_config)
            
            # 8. Calcular métricas finales
            metrics = {
                'output_dir': str(output_path),
                'training_time': (time.time() - start_time) / 60,  # minutos
                'final_loss': train_result.training_loss,
                'total_steps': train_result.global_step,
                'model_name': self.model_name,
                'lora_rank': lora_config.r,
                'samples_trained': len(train_dataset)
            }
            
            # Limpiar memoria
            self._cleanup()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            self._cleanup()
            raise
    
    def _load_model_and_tokenizer(self):
        """Carga el modelo y tokenizer"""
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        # Añadir tokens especiales si no existen
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración para carga eficiente
        if self.device.type == "cuda":
            # Intentar cargar en 8bit para ahorrar memoria
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Modelo cargado en 8-bit")
            except:
                # Si falla, cargar normal
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Modelo cargado en FP16")
        else:
            # CPU
            try:
                # Intentar cargar modelo estándar
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except OSError as e:
                # Si falla, intentar cargar modelo guardado por componentes
                if "no file named pytorch_model.bin" in str(e):
                    logger.info("Modelo guardado por componentes detectado, cargando...")
                    self.model = self._load_component_model()
                else:
                    raise e
            
            self.model = self.model.to(self.device)
        
        # Preparar para entrenamiento
        self.model.enable_input_require_grads()
        
    def _prepare_datasets(self, training_data: List[Dict[str, str]], 
                         data_config: 'DataConfig') -> tuple:
        """Prepara los datasets de entrenamiento y evaluación"""
        
        # Tokenizar datos
        tokenized_data = []
        
        for sample in tqdm(training_data, desc="Tokenizando"):
            # Formatear texto según el template
            if 'text' in sample:
                text = sample['text']
            elif 'instruction' in sample and 'response' in sample:
                text = data_config.instruction_template.format(**sample)
            else:
                continue
            
            # Tokenizar
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=data_config.max_length,
                return_tensors=None
            )
            
            # Añadir labels (igual que input_ids para LM)
            encoded['labels'] = encoded['input_ids'].copy()
            
            tokenized_data.append(encoded)
        
        # Crear dataset
        dataset = Dataset.from_list(tokenized_data)
        
        # Split train/eval
        if data_config.eval_split_ratio > 0:
            split = dataset.train_test_split(
                test_size=data_config.eval_split_ratio,
                seed=data_config.seed
            )
            train_dataset = split['train']
            eval_dataset = split['test']
        else:
            train_dataset = dataset
            eval_dataset = None
        
        logger.info(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def _setup_lora(self, lora_config: 'LoRAConfig'):
        """Configura LoRA en el modelo"""
        # Preparar modelo para entrenamiento int8
        if hasattr(self.model, 'is_loaded_in_8bit') and self.model.is_loaded_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Detectar módulos disponibles si no se especificaron
        if not lora_config.target_modules:
            lora_config.target_modules = self._detect_target_modules()
        
        logger.info(f"🎯 Módulos objetivo para LoRA: {lora_config.target_modules}")
        logger.info(f"📊 Total de módulos objetivo: {len(lora_config.target_modules)}")
        
        # Crear configuración LoRA de PEFT
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_config.target_modules,
            modules_to_save=lora_config.modules_to_save
        )
        
        logger.info(f"🔧 Configuración LoRA creada: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        
        # Aplicar LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Imprimir información
        self.model.print_trainable_parameters()
        
    # En lora_trainer.py, modifica el método _create_training_args, alrededor de la línea 273:

    def _create_training_args(self, training_config: 'TrainingConfig', 
                        dataset_size: int) -> TrainingArguments:
        """Crea argumentos de entrenamiento - versión compatible"""
        # Calcular pasos totales
        steps_per_epoch = dataset_size // training_config.per_device_train_batch_size
        total_steps = steps_per_epoch * training_config.num_train_epochs
        
        # Crear nombre único para run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.model_name}_lora_{timestamp}"
        
        # Directorio de salida
        output_dir = self.output_dir / run_name
        
        # Argumentos compatibles con versiones antiguas y nuevas
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            weight_decay=training_config.weight_decay,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            fp16=training_config.fp16 and self.device.type == "cuda",
            gradient_checkpointing=training_config.gradient_checkpointing,
            run_name=run_name,
            do_eval=False,  # Simplificado para evitar problemas
            push_to_hub=False,
            remove_unused_columns=False,  # Evitar problemas con columnas del dataset
        )
        
        return args
    
    def _create_trainer(self, training_args: TrainingArguments,
                       train_dataset: Dataset,
                       eval_dataset: Optional[Dataset],
                       data_config: 'DataConfig') -> Trainer:
        """Crea el trainer"""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
            pad_to_multiple_of=8  # Optimización
        )
        
        # Callbacks
        callbacks = [
            ProgressCallback(training_args.max_steps or 
                           (len(train_dataset) // training_args.per_device_train_batch_size) * 
                           training_args.num_train_epochs)
        ]
        
        #if eval_dataset is not None:
        #    callbacks.append(
        #        EarlyStoppingCallback(
        #            early_stopping_patience=3,
        #            early_stopping_threshold=0.001
        #        )
        #    )
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
            compute_metrics=None  # Por ahora sin métricas custom
        )
        
        return trainer
    
    def _save_model(self, trainer: Trainer, training_config: 'TrainingConfig') -> Path:
        """Guarda el modelo entrenado"""
        # Guardar modelo y tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(trainer.args.output_dir)
        
        # Guardar información adicional
        info = {
            'base_model': self.model_name,
            'base_model_path': str(self.model_path),
            'training_completed': datetime.now().isoformat(),
            'final_checkpoint': trainer.state.global_step,
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'training_loss': trainer.state.log_history[-1].get('loss', None) if trainer.state.log_history else None
        }
        
        info_path = Path(trainer.args.output_dir) / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Modelo guardado en: {trainer.args.output_dir}")
        
        return Path(trainer.args.output_dir)
    
    def _cleanup(self):
        """Limpia memoria"""
        if self.model:
            del self.model
        if self.peft_model:
            del self.peft_model
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_component_model(self):
        """Carga un modelo guardado por componentes"""
        import json
        from pathlib import Path
        
        model_dir = Path(self.model_path)
        config_path = model_dir / 'config.json'
        
        if not config_path.exists():
            raise ValueError(f"No se encontró config.json en {model_dir}")
        
        # Cargar configuración
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Crear modelo desde configuración
        from transformers import GPT2Config
        
        # Convertir diccionario a objeto de configuración
        if isinstance(config, dict):
            config = GPT2Config(**config)
        
        model_class = AutoModelForCausalLM
        model = model_class.from_config(config)
        
        # Cargar parámetros guardados por componentes
        param_files = list(model_dir.glob('*.pt'))
        if not param_files:
            raise ValueError(f"No se encontraron archivos de parámetros .pt en {model_dir}")
        
        # Crear state_dict
        state_dict = {}
        for param_file in param_files:
            param_name = param_file.stem
            param_data = torch.load(param_file, map_location='cpu')
            state_dict[param_name] = param_data
        
        # Cargar parámetros en el modelo
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Claves faltantes: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Claves inesperadas: {len(unexpected_keys)}")
        
        logger.info(f"Modelo cargado desde componentes: {len(param_files)} parámetros")
        return model
    
    def _detect_target_modules(self):
        """Detecta automáticamente los módulos objetivo para LoRA en múltiples arquitecturas"""
        target_modules = []
        
        # Detectar arquitectura del modelo
        model_type = getattr(self.model.config, 'model_type', 'unknown').lower()
        logger.info(f"🔍 Detectando módulos para arquitectura: {model_type}")
        
        # Módulos objetivo por arquitectura
        architecture_modules = {
            'gpt2': {
                'attention': ['attn.c_attn', 'attn.c_proj'],
                'mlp': ['mlp.c_fc', 'mlp.c_proj'],
                'pattern': 'transformer.h.{}.{}'
            },
            'llama': {
                'attention': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
                'mlp': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
                'pattern': 'model.layers.{}.{}'
            },
            'bert': {
                'attention': ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense'],
                'mlp': ['intermediate.dense', 'output.dense'],
                'pattern': 'encoder.layer.{}.{}'
            },
            'bart': {
                'attention': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj'],
                'mlp': ['fc1', 'fc2'],
                'pattern': 'encoder.layers.{}.{}'
            },
            't5': {
                'attention': ['SelfAttention.q', 'SelfAttention.k', 'SelfAttention.v', 'SelfAttention.o'],
                'mlp': ['DenseReluDense.wi', 'DenseReluDense.wo'],
                'pattern': 'encoder.block.{}.layer.{}'
            }
        }
        
        # Obtener configuración para la arquitectura
        arch_config = architecture_modules.get(model_type, architecture_modules['gpt2'])
        
        # Detectar número de capas
        num_layers = self._detect_num_layers(model_type)
        logger.info(f"📊 Detectadas {num_layers} capas")
        
        # Generar módulos objetivo
        for layer_idx in range(num_layers):
            for module_type, module_names in arch_config.items():
                if module_type == 'attention':
                    for module_name in module_names:
                        if '{}' in arch_config['pattern']:
                            target_name = arch_config['pattern'].format(layer_idx, module_name)
                        else:
                            target_name = f"layer_{layer_idx}.{module_name}"
                        target_modules.append(target_name)
                elif module_type == 'mlp':
                    for module_name in module_names:
                        if '{}' in arch_config['pattern']:
                            target_name = arch_config['pattern'].format(layer_idx, module_name)
                        else:
                            target_name = f"layer_{layer_idx}.{module_name}"
                        target_modules.append(target_name)
        
        # Si no se encontraron módulos, usar detección genérica
        if not target_modules:
            logger.info("🔄 Usando detección genérica de módulos...")
            target_modules = self._generic_module_detection()
        
        # Verificar que los módulos existen en el modelo
        verified_modules = []
        for module_name in target_modules:
            try:
                # Buscar el módulo en el modelo
                for name, module in self.model.named_modules():
                    if module_name in name and isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                        verified_modules.append(name)
                        break
            except Exception as e:
                logger.debug(f"⚠️ Error verificando módulo {module_name}: {e}")
                continue
        
        if not verified_modules:
            logger.warning("⚠️ No se pudieron verificar módulos, usando lista hardcodeada")
            verified_modules = self._fallback_modules(model_type, num_layers)
        
        logger.info(f"🎯 Módulos objetivo finales: {len(verified_modules)}")
        logger.info(f"  Primeros 5 módulos:")
        for i, module in enumerate(verified_modules[:5], 1):
            logger.info(f"    {i}. {module}")
        if len(verified_modules) > 5:
            logger.info(f"  ... y {len(verified_modules) - 5} más")
        
        return verified_modules
    
    def _detect_num_layers(self, model_type: str) -> int:
        """Detecta el número de capas del modelo"""
        try:
            if hasattr(self.model.config, 'n_layer'):
                return self.model.config.n_layer
            elif hasattr(self.model.config, 'num_hidden_layers'):
                return self.model.config.num_hidden_layers
            elif hasattr(self.model.config, 'num_layers'):
                return self.model.config.num_layers
            else:
                # Contar capas manualmente
                layer_count = 0
                for name, _ in self.model.named_modules():
                    if any(keyword in name for keyword in ['h.', 'layers.', 'block.']):
                        layer_count = max(layer_count, int(name.split('.')[1]) + 1)
                return layer_count if layer_count > 0 else 12
        except:
            return 12  # Valor por defecto
    
    def _generic_module_detection(self):
        """Detección genérica de módulos para arquitecturas desconocidas"""
        target_modules = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                # Buscar módulos de atención y MLP
                if any(keyword in name.lower() for keyword in ['attn', 'attention', 'mlp', 'ffn', 'proj']):
                    target_modules.append(name)
        
        return target_modules
    
    def _fallback_modules(self, model_type: str, num_layers: int):
        """Módulos de respaldo si falla la detección automática"""
        if model_type == 'gpt2':
            fallback = []
            for i in range(num_layers):
                fallback.extend([
                    f'transformer.h.{i}.attn.c_attn',
                    f'transformer.h.{i}.attn.c_proj',
                    f'transformer.h.{i}.mlp.c_fc',
                    f'transformer.h.{i}.mlp.c_proj'
                ])
            return fallback
        else:
            # Para otras arquitecturas, usar módulos genéricos
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']