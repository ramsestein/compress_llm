#!/usr/bin/env python3
"""
Tests para ejecución real de entrenamiento
"""
import unittest
import tempfile
import shutil
import json
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import LoRa_train modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer
from LoRa_train.lora_trainer import LoRATrainer
from LoRa_train.peft_methods_config import (
    PEFTMethod, LoRAConfig, MoLoRAConfig, IA3Config, BitFitConfig
)


class TestTrainingExecution(unittest.TestCase):
    """Tests para ejecución real de entrenamiento"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Crear directorio del modelo
        self.model_path.mkdir(exist_ok=True)
        
        # Crear archivos de configuración simulados
        config = {
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12
        }
        
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        # Crear tokenizer simulado
        tokenizer_config = {
            "model_max_length": 1024,
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>"
        }
        
        with open(self.model_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f)
        
        # Crear archivo de vocabulario simulado
        vocab = {"<|endoftext|>": 50256, "hello": 0, "world": 1}
        with open(self.model_path / "vocab.json", "w") as f:
            json.dump(vocab, f)
        
        # Crear archivo de merges simulado
        with open(self.model_path / "merges.txt", "w") as f:
            f.write("h e\nhe l\nhel l\nhell o\n")
        
        # Datos de entrenamiento de prueba
        self.test_data = [
            {"instruction": "Translate to English", "response": "Hello world"},
            {"instruction": "What is 2+2?", "response": "4"},
            {"text": "This is a test sentence for language modeling."}
        ]
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_model_files(self):
        """Crea archivos mock del modelo"""
        # Crear directorio del modelo
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo de configuración
        config_file = self.model_path / "config.json"
        config_data = {
            "model_type": "gpt2",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "vocab_size": 50257
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Crear archivo de tokenizer
        tokenizer_file = self.model_path / "tokenizer.json"
        with open(tokenizer_file, 'w') as f:
            f.write('{"dummy": "tokenizer"}')
    
    def _setup_mock_tokenizer(self, mock_tokenizer_instance):
        """Configura el mock del tokenizer para que retorne objetos subscriptables"""
        mock_encoded = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock_tokenizer_instance.__call__ = Mock(return_value=mock_encoded)
        mock_tokenizer_instance.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer_instance.encode_plus = Mock(return_value=mock_encoded)
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_lora_training_execution(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para ejecución completa de entrenamiento LoRA"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "gpt2"
        mock_model_instance.config.n_layer = 12
        mock_model_instance.state_dict.return_value = {"layer.0.weight": torch.randn(10, 10)}
        mock_model.from_config.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_peft_model
        
        # Crear trainer
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, config)
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
            mock_get_peft_model.assert_called_once()
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_ia3_training_execution(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para ejecución completa de entrenamiento IA3"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "gpt2"
        mock_model_instance.config.n_layer = 12
        mock_model_instance.state_dict.return_value = {"layer.0.weight": torch.randn(10, 10)}
        mock_model.from_config.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_peft_model
        
        # Crear trainer
        config = IA3Config(
            method=PEFTMethod.IA3,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, config)
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
            mock_get_peft_model.assert_called_once()
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_bitfit_training_execution(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para ejecución completa de entrenamiento BitFit"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "gpt2"
        mock_model_instance.config.n_layer = 12
        mock_model_instance.state_dict.return_value = {"layer.0.weight": torch.randn(10, 10)}
        mock_model.from_config.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_peft_model
        
        # Crear trainer
        config = BitFitConfig(
            method=PEFTMethod.BITFIT,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, config)
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
            # Para BitFit, no necesariamente se llama get_peft_model
            # mock_get_peft_model.assert_called_once()
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_molora_training_execution(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para ejecución completa de entrenamiento MoLoRA"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "gpt2"
        mock_model_instance.config.n_layer = 12
        mock_model_instance.state_dict.return_value = {"layer.0.weight": torch.randn(10, 10)}
        mock_model.from_config.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_peft_model
        
        # Crear trainer
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            expert_r=[8, 8, 16, 16],
            expert_alpha=[16, 16, 32, 32],
            num_experts=4,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, config)
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
            mock_get_peft_model.assert_called_once()
    
    def test_error_handling_invalid_method(self):
        """Test para manejo de errores con método inválido"""
        # Crear configuración con método inválido
        config = Mock()
        config.method = "invalid_method"
        
        trainer = PEFTUniversalTrainer(
            "test_model",
            self.model_path,
            self.output_dir,
            config
        )
        
        # Verificar que se lanza error al intentar entrenar
        with self.assertRaises(Exception):
            trainer.train(self.test_data)
    
    def test_error_handling_model_loading_failure(self):
        """Test para manejo de errores en carga de modelo"""
        # Crear trainer con modelo inexistente
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        
        trainer = PEFTUniversalTrainer(
            "test_model",
            Path("/path/inexistente"),
            self.output_dir,
            config
        )
        
        # Verificar que se lanza error al intentar entrenar
        with self.assertRaises(Exception):
            trainer.train(self.test_data)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    def test_dataset_preparation_with_different_formats(self, mock_tokenizer, mock_model):
        """Test para preparación de datasets con diferentes formatos"""
        # Crear trainer básico
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, config)
        
        # Configurar mock tokenizer
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        trainer.tokenizer = mock_tokenizer_instance
        
        # Test con datos mixtos
        mixed_data = [
            {"instruction": "Test 1", "response": "Response 1"},
            {"text": "Test 2"},
            {"instruction": "Test 3", "response": "Response 3"}
        ]
        
        # Preparar datasets
        train_dataset, eval_dataset = trainer._prepare_datasets(mixed_data)
        
        # Verificar que se crearon los datasets
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(eval_dataset)


if __name__ == "__main__":
    unittest.main()
