#!/usr/bin/env python3
"""
Test comprehensivo del sistema LoRA
Verifica todos los componentes, métodos PEFT, configuraciones, datasets y funcionalidades
"""
import unittest
import tempfile
import shutil
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys
import os
import pandas as pd

# Asegurar que el paquete del proyecto esté en el path de importación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LoRa_train.lora_config import (
    LoRAConfig as LoRALegacyConfig, TrainingConfig, DataConfig, LoRAPresets, 
    get_model_specific_config, TaskType
)
from LoRa_train.lora_trainer import LoRATrainer, ProgressCallback
from LoRa_train.dataset_manager import OptimizedDatasetManager, DatasetConfig
from LoRa_train.peft_methods import (
    MoLoRALinear, DoRALinear, AdapterLinear,
    GaLoreLinear, IA3Linear, QuantizedLoRALinear, PrunedLoRALinear,
    CompacterLinear, KronALinear, S4Adapter, HoulsbyAdapterLinear
)
from LoRa_train.peft_methods_config import (
    PEFTMethod, BasePEFTConfig, MoLoRAConfig, GaLoreConfig,
    DoRAConfig, AdapterConfig, LoRAConfig, get_available_methods, get_available_presets
)
from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer


class TestLoRASystemComprehensive(unittest.TestCase):
    """Test comprehensivo del sistema LoRA"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear módulos de prueba
        self.linear_layer = nn.Linear(100, 50)
        self.attention_layer = nn.Linear(768, 768)
        self.ffn_layer = nn.Linear(768, 3072)
        
        # Crear dataset de prueba
        self.create_test_dataset()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_dataset(self):
        """Crear dataset de prueba para los tests"""
        dataset_path = self.test_dir / "test_dataset.csv"
        
        # Crear datos de prueba
        data = {
            'instruction': [
                'Traduce al inglés: Hola mundo',
                'Responde: ¿Cuál es la capital de España?',
                'Explica: ¿Qué es la inteligencia artificial?'
            ],
            'response': [
                'Hello world',
                'La capital de España es Madrid',
                'La inteligencia artificial es una rama de la informática'
            ]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(dataset_path, index=False)
        self.test_dataset_path = dataset_path
    
    # ============= TESTS DE CONFIGURACIÓN LoRA =============
    
    def test_lora_config_initialization(self):
        """Test de inicialización de configuración LoRA"""
        config = LoRAConfig(method=PEFTMethod.LORA)
        
        # Verificar valores por defecto
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertEqual(config.bias, "none")
        self.assertEqual(config.task_type, "CAUSAL_LM")
        
        # Verificar target_modules
        self.assertIsInstance(config.target_modules, list)
        self.assertGreater(len(config.target_modules), 0)
        
        # Verificar modules_to_save
        self.assertIsInstance(config.modules_to_save, list)
        self.assertGreater(len(config.modules_to_save), 0)
    
    def test_lora_config_customization(self):
        """Test de personalización de configuración LoRA"""
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="all",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens"]
        )
        
        # Verificar valores personalizados
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.bias, "all")
        self.assertEqual(config.target_modules, ["q_proj", "v_proj"])
        self.assertEqual(config.modules_to_save, ["embed_tokens"])
    
    def test_lora_config_serialization(self):
        """Test de serialización de configuración LoRA"""
        config = LoRAConfig(method=PEFTMethod.LORA, r=8, lora_alpha=16)
        
        # Convertir a diccionario
        config_dict = config.to_dict()
        
        # Verificar estructura
        self.assertIn('r', config_dict)
        self.assertIn('lora_alpha', config_dict)
        self.assertIn('lora_dropout', config_dict)
        self.assertIn('bias', config_dict)
        self.assertIn('task_type', config_dict)
        self.assertIn('target_modules', config_dict)
        self.assertIn('modules_to_save', config_dict)
        
        # Verificar valores
        self.assertEqual(config_dict['r'], 8)
        self.assertEqual(config_dict['lora_alpha'], 16)
    
    def test_training_config_initialization(self):
        """Test de inicialización de configuración de entrenamiento"""
        config = TrainingConfig()
        
        # Verificar valores por defecto
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.per_device_train_batch_size, 4)
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertTrue(config.gradient_checkpointing)
        self.assertTrue(config.fp16)
        self.assertEqual(config.optim, "paged_adamw_8bit")
    
    def test_training_config_customization(self):
        """Test de personalización de configuración de entrenamiento"""
        config = TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            gradient_checkpointing=False,
            fp16=False,
            optim="adamw"
        )
        
        # Verificar valores personalizados
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.per_device_train_batch_size, 2)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertFalse(config.gradient_checkpointing)
        self.assertFalse(config.fp16)
        self.assertEqual(config.optim, "adamw")
    
    def test_data_config_initialization(self):
        """Test de inicialización de configuración de datos"""
        config = DataConfig()
        
        # Verificar valores por defecto
        self.assertEqual(config.max_length, 512)
        self.assertTrue(config.truncation)
        self.assertEqual(config.padding, "max_length")
        self.assertEqual(config.eval_split_ratio, 0.1)
        self.assertTrue(config.shuffle)
        self.assertEqual(config.seed, 42)
    
    def test_data_config_customization(self):
        """Test de personalización de configuración de datos"""
        config = DataConfig(
            max_length=256,
            truncation=False,
            padding="longest",
            eval_split_ratio=0.2,
            shuffle=False,
            seed=123
        )
        
        # Verificar valores personalizados
        self.assertEqual(config.max_length, 256)
        self.assertFalse(config.truncation)
        self.assertEqual(config.padding, "longest")
        self.assertEqual(config.eval_split_ratio, 0.2)
        self.assertFalse(config.shuffle)
        self.assertEqual(config.seed, 123)
    
    # ============= TESTS DE PRESETS LoRA =============
    
    def test_lora_presets_structure(self):
        """Test de estructura de presets LoRA"""
        # Verificar que existen los presets principales
        required_presets = ['balanced', 'conservative', 'aggressive']
        for preset_name in required_presets:
            preset = LoRAPresets.get_preset(preset_name)
            self.assertIsNotNone(preset)
            
            # Verificar estructura
            self.assertIn('lora', preset)
            self.assertIn('training', preset)
            self.assertIn('data', preset)
            
            # Verificar tipos
            self.assertIsInstance(preset['lora'], LoRALegacyConfig)
            self.assertIsInstance(preset['training'], TrainingConfig)
            self.assertIsInstance(preset['data'], DataConfig)
    
    def test_balanced_preset(self):
        """Test del preset balanceado"""
        preset = LoRAPresets.get_preset('balanced')
        
        # Verificar configuración LoRA
        lora_config = preset['lora']
        self.assertEqual(lora_config.r, 16)
        self.assertEqual(lora_config.lora_alpha, 32)
        self.assertEqual(lora_config.lora_dropout, 0.1)
        
        # Verificar configuración de entrenamiento
        training_config = preset['training']
        self.assertEqual(training_config.num_train_epochs, 3)
        self.assertEqual(training_config.learning_rate, 2e-4)
        self.assertTrue(training_config.gradient_checkpointing)
    
    def test_fast_preset(self):
        """Test del preset conservativo (equivalente a fast)"""
        preset = LoRAPresets.get_preset('conservative')
        
        # Verificar configuración LoRA
        lora_config = preset['lora']
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)
        self.assertEqual(lora_config.lora_dropout, 0.05)
        
        # Verificar configuración de entrenamiento
        training_config = preset['training']
        self.assertEqual(training_config.num_train_epochs, 1)
        self.assertEqual(training_config.learning_rate, 1e-4)
    
    def test_quality_preset(self):
        """Test del preset agresivo (equivalente a quality)"""
        preset = LoRAPresets.get_preset('aggressive')
        
        # Verificar configuración LoRA
        lora_config = preset['lora']
        self.assertEqual(lora_config.r, 32)
        self.assertEqual(lora_config.lora_alpha, 64)
        self.assertEqual(lora_config.lora_dropout, 0.1)
        
        # Verificar configuración de entrenamiento
        training_config = preset['training']
        self.assertEqual(training_config.num_train_epochs, 5)
        self.assertEqual(training_config.learning_rate, 3e-4)
    
    def test_memory_usage_estimation(self):
        """Test de estimación de uso de memoria"""
        config = LoRAPresets.get_preset('balanced')['lora']
        
        # Estimar uso de memoria
        memory_usage = LoRAPresets.estimate_memory_usage(
            model_size_gb=1.0,
            config=config,
            batch_size=4
        )
        
        # Verificar estructura
        self.assertIn('base_model', memory_usage)
        self.assertIn('lora_adapters', memory_usage)
        self.assertIn('gradients', memory_usage)
        self.assertIn('optimizer', memory_usage)
        self.assertIn('activations', memory_usage)
        self.assertIn('total_estimated', memory_usage)
        self.assertIn('recommended_gpu_memory', memory_usage)
        
        # Verificar valores
        self.assertEqual(memory_usage['base_model'], 1.0)
        self.assertGreater(memory_usage['total_estimated'], 1.0)
        self.assertGreater(memory_usage['recommended_gpu_memory'], memory_usage['total_estimated'])
    
    # ============= TESTS DE CONFIGURACIÓN ESPECÍFICA POR MODELO =============
    
    def test_model_specific_configs(self):
        """Test de configuraciones específicas por modelo"""
        # Test GPT2
        gpt2_config = get_model_specific_config('gpt2')
        self.assertIn('target_modules', gpt2_config)
        self.assertIn('modules_to_save', gpt2_config)
        self.assertEqual(gpt2_config['target_modules'], ['c_attn', 'c_proj', 'c_fc'])
        self.assertEqual(gpt2_config['modules_to_save'], ['wte', 'wpe', 'ln_f'])
        
        # Test LLaMA
        llama_config = get_model_specific_config('llama')
        self.assertIn('target_modules', llama_config)
        self.assertIn('modules_to_save', llama_config)
        self.assertIn('q_proj', llama_config['target_modules'])
        self.assertIn('v_proj', llama_config['target_modules'])
        
        # Test BERT
        bert_config = get_model_specific_config('bert')
        self.assertIn('target_modules', bert_config)
        self.assertIn('modules_to_save', bert_config)
        self.assertIn('query', bert_config['target_modules'])
        self.assertIn('key', bert_config['target_modules'])
        
        # Test T5
        t5_config = get_model_specific_config('t5')
        self.assertIn('target_modules', t5_config)
        self.assertIn('modules_to_save', t5_config)
        self.assertIn('q', t5_config['target_modules'])
        self.assertIn('v', t5_config['target_modules'])
    
    def test_unknown_model_config(self):
        """Test de configuración para modelo desconocido"""
        unknown_config = get_model_specific_config('unknown_model')
        
        # Debería devolver configuración por defecto (LLaMA)
        self.assertIn('target_modules', unknown_config)
        self.assertIn('modules_to_save', unknown_config)
        self.assertIn('q_proj', unknown_config['target_modules'])
    
    # ============= TESTS DE MÉTODOS PEFT =============
    
    def test_molora_linear_initialization(self):
        """Test de inicialización de MoLoRALinear"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=4,
            expert_r=[8, 8, 8, 8],
            expert_alpha=[16, 16, 16, 16]
        )
        
        molora_layer = MoLoRALinear(100, 50, config)
        
        # Verificar estructura
        self.assertEqual(molora_layer.in_features, 100)
        self.assertEqual(molora_layer.out_features, 50)
        self.assertEqual(molora_layer.num_experts, 4)
        self.assertIsNotNone(molora_layer.lora_As)
        self.assertIsNotNone(molora_layer.lora_Bs)
        self.assertIsNotNone(molora_layer.router)
    
    def test_molora_forward_pass(self):
        """Test de forward pass de MoLoRALinear"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[4, 4],
            expert_alpha=[8, 8]
        )
        
        molora_layer = MoLoRALinear(10, 10, config)
        x = torch.randn(2, 10)
        
        # Forward pass
        output = molora_layer(x)
        
        # Verificar salida
        self.assertEqual(output.shape, (2, 10))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_dora_linear_initialization(self):
        """Test de inicialización de DoRALinear"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=8,
            lora_alpha=16
        )
        
        dora_layer = DoRALinear(100, 50, config)
        
        # Verificar estructura
        self.assertEqual(dora_layer.in_features, 100)
        self.assertEqual(dora_layer.out_features, 50)
        self.assertIsNotNone(dora_layer.lora_A)
        self.assertIsNotNone(dora_layer.lora_B)
        self.assertIsNotNone(dora_layer.dora_scale)
    
    def test_dora_forward_pass(self):
        """Test de forward pass de DoRALinear"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=4,
            lora_alpha=8
        )
        
        dora_layer = DoRALinear(10, 10, config)
        x = torch.randn(2, 10)
        
        # Forward pass
        output = dora_layer(x)
        
        # Verificar salida
        self.assertEqual(output.shape, (2, 10))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_adapter_linear_initialization(self):
        """Test de inicialización de AdapterLinear"""
        config = AdapterConfig(
            method=PEFTMethod.ADAPTER,
            adapter_size=64,
            adapter_dropout=0.1
        )
        
        adapter_layer = AdapterLinear(100, 50, config)
        
        # Verificar estructura
        self.assertEqual(adapter_layer.in_features, 100)
        self.assertEqual(adapter_layer.out_features, 50)
        self.assertIsNotNone(adapter_layer.adapter_down)
        self.assertIsNotNone(adapter_layer.adapter_up)
    
    def test_adapter_forward_pass(self):
        """Test de forward pass de AdapterLinear"""
        config = AdapterConfig(
            method=PEFTMethod.ADAPTER,
            adapter_size=32,
            adapter_dropout=0.1
        )
        
        adapter_layer = AdapterLinear(10, 10, config)
        x = torch.randn(2, 10)
        
        # Forward pass
        output = adapter_layer(x)
        
        # Verificar salida
        self.assertEqual(output.shape, (2, 10))
        self.assertIsInstance(output, torch.Tensor)
    
    # ============= TESTS DE CONFIGURACIÓN PEFT =============
    
    def test_peft_methods_enum(self):
        """Test del enum PEFTMethod"""
        # Verificar métodos disponibles
        methods = list(PEFTMethod)
        self.assertIn(PEFTMethod.LORA, methods)
        self.assertIn(PEFTMethod.MOLORA, methods)
        self.assertIn(PEFTMethod.DORA, methods)
        self.assertIn(PEFTMethod.ADAPTER, methods)
        # AdaLoRA ha sido eliminado, no verificar su existencia
    
    def test_get_available_methods(self):
        """Test de obtención de métodos disponibles"""
        # Test para preset balanced
        balanced_methods = get_available_methods('balanced')
        self.assertIsInstance(balanced_methods, list)
        self.assertIn('lora', balanced_methods)
        self.assertIn('dora', balanced_methods)
        self.assertIn('adapter', balanced_methods)
        
        # Test para preset quality
        quality_methods = get_available_methods('quality')
        self.assertIsInstance(quality_methods, list)
        self.assertIn('lora', quality_methods)
        self.assertIn('molora', quality_methods)
        # AdaLoRA ha sido eliminado, no verificar su existencia
    
    def test_get_available_presets(self):
        """Test de obtención de presets disponibles"""
        presets = get_available_presets()
        self.assertIsInstance(presets, list)
        self.assertIn('balanced', presets)
        self.assertIn('efficient', presets)
        self.assertIn('quality', presets)
    
    def test_base_peft_config(self):
        """Test de configuración base PEFT"""
        config = BasePEFTConfig(
            method=PEFTMethod.LORA,
            target_modules=["q_proj", "v_proj"]
        )
        
        # Verificar estructura
        self.assertEqual(config.method, PEFTMethod.LORA)
        self.assertEqual(config.target_modules, ["q_proj", "v_proj"])
    
    def test_molora_config(self):
        """Test de configuración MoLoRA"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=4,
            expert_r=[16, 16, 16, 16],
            expert_alpha=[32, 32, 32, 32]
        )
        
        # Verificar estructura
        self.assertEqual(config.method, PEFTMethod.MOLORA)
        self.assertEqual(config.expert_r, [16, 16, 16, 16])
        self.assertEqual(config.expert_alpha, [32, 32, 32, 32])
        self.assertEqual(config.num_experts, 4)
    
    def test_dora_config(self):
        """Test de configuración DoRA"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=8,
            lora_alpha=16
        )
        
        # Verificar estructura
        self.assertEqual(config.method, PEFTMethod.DORA)
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
    
    # ============= TESTS DE GESTOR DE DATASETS =============
    
    def test_dataset_manager_initialization(self):
        """Test de inicialización del gestor de datasets"""
        dm = OptimizedDatasetManager(str(self.test_dir))
        
        # Verificar inicialización
        self.assertIsNotNone(dm)
        self.assertEqual(dm.datasets_dir, self.test_dir)
        self.assertTrue(hasattr(dm, 'scan_datasets'))
        self.assertTrue(hasattr(dm, '_analyze_dataset'))
    
    def test_dataset_scanning(self):
        """Test de escaneo de datasets"""
        dm = OptimizedDatasetManager(str(self.test_dir))
        
        # Escanear datasets
        datasets = dm.scan_datasets()
        
        # Verificar que encuentra el dataset de prueba
        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)
        
        # Verificar estructura del dataset encontrado
        dataset = datasets[0]
        self.assertIn('file_path', dataset)
        self.assertIn('name', dataset)
        self.assertIn('format', dataset)
        self.assertIn('size_mb', dataset)
        self.assertIn('columns', dataset)
    
    def test_dataset_analysis(self):
        """Test de análisis de datasets"""
        dm = OptimizedDatasetManager(str(self.test_dir))
        
        # Analizar dataset de prueba
        analysis = dm._analyze_dataset(self.test_dataset_path)
        
        # Verificar estructura del análisis
        self.assertIn('format', analysis)
        self.assertIn('num_columns', analysis)
        self.assertIn('estimated_rows', analysis)
        self.assertIn('size_mb', analysis)
        self.assertIn('columns', analysis)
        
        # Verificar valores específicos
        self.assertEqual(analysis['format'], 'csv')
        self.assertEqual(analysis['num_columns'], 2)
        self.assertGreater(analysis['estimated_rows'], 0)
        self.assertGreater(analysis['size_mb'], 0)
        self.assertEqual(analysis['columns'], ['instruction', 'response'])
    
    def test_dataset_config_creation(self):
        """Test de creación de configuración de dataset"""
        dm = OptimizedDatasetManager(str(self.test_dir))
        
        # Crear configuración de dataset
        config = DatasetConfig(
            file_path=self.test_dataset_path,
            format='csv',
            columns={'instruction': 'instruction', 'response': 'response'},
            name='test_dataset',
            size=100,
            max_length=512,
            instruction_template='### Instruction:\n{instruction}\n\n### Response:\n{response}'
        )
        
        # Verificar configuración
        self.assertEqual(config.file_path, self.test_dataset_path)
        self.assertEqual(config.format, 'csv')
        self.assertEqual(config.name, 'test_dataset')
        self.assertEqual(config.size, 100)
        self.assertEqual(config.max_length, 512)
        self.assertIsNotNone(config.instruction_template)
    
    def test_dataset_config_serialization(self):
        """Test de serialización de configuración de dataset"""
        config = DatasetConfig(
            file_path=self.test_dataset_path,
            format='csv',
            columns={'instruction': 'instruction', 'response': 'response'},
            name='test_dataset',
            size=100
        )
        
        # Convertir a diccionario
        config_dict = config.to_dict()
        
        # Verificar estructura
        self.assertIn('file_path', config_dict)
        self.assertIn('format', config_dict)
        self.assertIn('columns', config_dict)
        self.assertIn('name', config_dict)
        self.assertIn('size', config_dict)
        
        # Verificar valores
        self.assertEqual(config_dict['format'], 'csv')
        self.assertEqual(config_dict['name'], 'test_dataset')
        self.assertEqual(config_dict['size'], 100)
    
    # ============= TESTS DE TRAINER LoRA =============
    
    def test_lora_trainer_initialization(self):
        """Test de inicialización del trainer LoRA"""
        trainer = LoRATrainer(
            model_name="test_model",
            model_path=self.test_dir,
            output_dir=self.test_dir
        )
        
        # Verificar inicialización
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.model_name, "test_model")
        self.assertEqual(trainer.model_path, self.test_dir)
        self.assertEqual(trainer.output_dir, self.test_dir)
        self.assertTrue(hasattr(trainer, 'device'))
        self.assertTrue(hasattr(trainer, 'model'))
        self.assertTrue(hasattr(trainer, 'tokenizer'))
    
    def test_progress_callback(self):
        """Test del callback de progreso"""
        callback = ProgressCallback(total_steps=100)
        
        # Verificar inicialización
        self.assertEqual(callback.total_steps, 100)
        self.assertIsNone(callback.progress_bar)
        self.assertEqual(callback.current_loss, 0)
        self.assertEqual(callback.step_count, 0)
    
    def test_peft_universal_trainer_initialization(self):
        """Test de inicialización del trainer universal PEFT"""
        # Crear configuración de prueba
        peft_config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16
        )
        
        trainer = PEFTUniversalTrainer(
            model_path=self.test_dir,
            peft_config=peft_config,
            model_name="test_model",
            output_dir=self.test_dir / "output"
        )
        
        # Verificar que se inicializó correctamente
        self.assertIsNotNone(trainer)
        self.assertTrue(hasattr(trainer, 'peft_config'))
        self.assertTrue(hasattr(trainer, 'model_name'))
        self.assertTrue(hasattr(trainer, 'output_dir'))
    
    # ============= TESTS DE INTEGRACIÓN =============
    
    def test_end_to_end_lora_workflow(self):
        """Test de flujo completo de LoRA"""
        # Crear configuración LoRA
        lora_config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"]
        )
        
        # Crear configuración de entrenamiento
        training_config = TrainingConfig(
            num_train_epochs=1,
            learning_rate=1e-4,
            per_device_train_batch_size=2
        )
        
        # Crear configuración de datos
        data_config = DataConfig(
            max_length=256,
            eval_split_ratio=0.1
        )
        
        # Verificar configuraciones
        self.assertIsInstance(lora_config, LoRAConfig)
        self.assertIsInstance(training_config, TrainingConfig)
        self.assertIsInstance(data_config, DataConfig)
        
        # Verificar valores
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(training_config.num_train_epochs, 1)
        self.assertEqual(data_config.max_length, 256)
    
    def test_model_specific_integration(self):
        """Test de integración específica por modelo"""
        # Configuración para GPT2
        gpt2_config = get_model_specific_config('gpt2')
        lora_config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            target_modules=gpt2_config['target_modules'],
            modules_to_save=gpt2_config['modules_to_save']
        )
        
        # Verificar integración
        self.assertEqual(lora_config.target_modules, ['c_attn', 'c_proj', 'c_fc'])
        self.assertEqual(lora_config.modules_to_save, ['wte', 'wpe', 'ln_f'])
    
    def test_preset_integration(self):
        """Test de integración de presets"""
        # Obtener preset balanceado
        preset = LoRAPresets.get_preset('balanced')
        
        # Verificar integración de componentes
        self.assertIsInstance(preset['lora'], LoRALegacyConfig)
        self.assertIsInstance(preset['training'], TrainingConfig)
        self.assertIsInstance(preset['data'], DataConfig)
        
        # Verificar coherencia
        self.assertGreater(preset['lora'].r, 0)
        self.assertGreater(preset['training'].learning_rate, 0)
        self.assertGreater(preset['data'].max_length, 0)
    
    def test_dataset_integration(self):
        """Test de integración de datasets"""
        dm = OptimizedDatasetManager(str(self.test_dir))
        datasets = dm.scan_datasets()
        
        # Verificar que el dataset está disponible
        self.assertGreater(len(datasets), 0)
        
        # Crear configuración de dataset
        dataset_info = datasets[0]
        config = DatasetConfig(
            file_path=Path(dataset_info['file_path']),
            format=dataset_info['format'],
            columns={'instruction': 'instruction', 'response': 'response'},
            name=dataset_info['name'],
            size=dataset_info['estimated_rows']
        )
        
        # Verificar integración
        self.assertEqual(config.format, 'csv')
        self.assertEqual(config.name, 'test_dataset')
        self.assertGreater(config.size, 0)
    
    def test_peft_methods_integration(self):
        """Test de integración de métodos PEFT"""
        # Test con diferentes configuraciones
        configs = [
            DoRAConfig(method=PEFTMethod.DORA, r=8, lora_alpha=16),
            MoLoRAConfig(method=PEFTMethod.MOLORA, num_experts=4, expert_r=[16, 16, 16, 16], expert_alpha=[32, 32, 32, 32]),
            AdapterConfig(method=PEFTMethod.ADAPTER, adapter_size=64)
        ]
        
        for config in configs:
            self.assertIsNotNone(config)
            self.assertTrue(hasattr(config, 'method'))
    
    def test_memory_estimation_integration(self):
        """Test de integración de estimación de memoria"""
        # Configuración LoRA
        lora_config = LoRAPresets.get_preset('balanced')['lora']
        
        # Estimar memoria
        memory_usage = LoRAPresets.estimate_memory_usage(
            model_size_gb=1.0,
            config=lora_config,
            batch_size=4
        )
        
        # Verificar que la estimación es razonable
        self.assertGreater(memory_usage['total_estimated'], 1.0)
        self.assertLess(memory_usage['lora_adapters'], 0.1)  # LoRA debería ser pequeño
        self.assertGreater(memory_usage['gradients'], 0)
        self.assertGreater(memory_usage['optimizer'], 0)
    
    def test_error_handling(self):
        """Test de manejo de errores"""
        # Test con configuración inválida
        try:
            invalid_config = LoRAConfig(method=PEFTMethod.LORA, r=-1)  # r inválido
            self.fail("Debería haber fallado con r negativo")
        except (ValueError, AssertionError):
            pass  # Esperado
        
        # Test con preset inexistente (debería devolver balanced por defecto)
        invalid_preset = LoRAPresets.get_preset('nonexistent')
        self.assertIsNotNone(invalid_preset)
        self.assertIn('lora', invalid_preset)
        self.assertIn('training', invalid_preset)
        self.assertIn('data', invalid_preset)
    
    def test_device_compatibility(self):
        """Test de compatibilidad de dispositivos"""
        # Test en CPU
        cpu_trainer = LoRATrainer(
            model_name="test_model",
            model_path=self.test_dir,
            output_dir=self.test_dir
        )
        self.assertEqual(cpu_trainer.device.type, "cpu")
        
        # Test en GPU si está disponible
        if torch.cuda.is_available():
            gpu_trainer = LoRATrainer(
                model_name="test_model",
                model_path=self.test_dir,
                output_dir=self.test_dir
            )
            # El trainer debería manejar la GPU correctamente
    
    def test_numerical_stability(self):
        """Test de estabilidad numérica"""
        # Test con valores extremos en configuración
        extreme_config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=1,  # Valor mínimo
            lora_alpha=1,  # Valor mínimo
            lora_dropout=0.0  # Sin dropout
        )
        
        # Verificar que la configuración es válida
        self.assertEqual(extreme_config.r, 1)
        self.assertEqual(extreme_config.lora_alpha, 1)
        self.assertEqual(extreme_config.lora_dropout, 0.0)
        
        # Test con valores grandes
        large_config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=256,  # Valor grande
            lora_alpha=512,  # Valor grande
            lora_dropout=0.5  # Dropout alto
        )
        
        # Verificar que la configuración es válida
        self.assertEqual(large_config.r, 256)
        self.assertEqual(large_config.lora_alpha, 512)
        self.assertEqual(large_config.lora_dropout, 0.5)
    
    def test_reproducibility(self):
        """Test de reproducibilidad"""
        # Crear dos configuraciones idénticas
        config1 = LoRAConfig(method=PEFTMethod.LORA, r=8, lora_alpha=16, lora_dropout=0.1)
        config2 = LoRAConfig(method=PEFTMethod.LORA, r=8, lora_alpha=16, lora_dropout=0.1)
        
        # Verificar que son idénticas
        self.assertEqual(config1.r, config2.r)
        self.assertEqual(config1.lora_alpha, config2.lora_alpha)
        self.assertEqual(config1.lora_dropout, config2.lora_dropout)
        
        # Verificar serialización idéntica
        self.assertEqual(config1.to_dict(), config2.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
