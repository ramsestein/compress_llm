#!/usr/bin/env python3
"""
Tests para PEFTUniversalTrainer
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
from LoRa_train.peft_methods_config import (
    PEFTMethod, LoRAConfig, MoLoRAConfig, GaLoreConfig, DoRAConfig,
    AdaLoRAConfig, BitFitConfig, IA3Config, PromptTuningConfig,
    AdapterConfig, QLoRAConfig
)


class TestPEFTUniversalTrainer(unittest.TestCase):
    """Tests para PEFTUniversalTrainer"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Crear archivos mock del modelo
        self._create_mock_model_files()
        
        # Configuración de prueba
        self.config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=8,
            lora_alpha=16,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        
        # Datos de prueba
        self.test_data = [
            {"instruction": "Test instruction 1", "response": "Test response 1"},
            {"instruction": "Test instruction 2", "response": "Test response 2"},
            {"instruction": "Test instruction 3", "response": "Test response 3"}
        ]
    
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
        # Crear un objeto real que pueda ser subscriptado
        class MockEncoded:
            def __init__(self):
                self.input_ids = [1, 2, 3]
                self.attention_mask = [1, 1, 1]
            
            def copy(self):
                return self
            
            def __getitem__(self, key):
                if key == 'input_ids':
                    return self.input_ids
                elif key == 'attention_mask':
                    return self.attention_mask
                else:
                    raise KeyError(key)
        
        mock_encoded = MockEncoded()
        # Asegurar que el mock retorne el objeto subscriptable
        mock_tokenizer_instance.__call__ = Mock(return_value=mock_encoded)
        mock_tokenizer_instance.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer_instance.encode_plus = Mock(return_value=mock_encoded)
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        # También mockear el método __call__ del mock para asegurar que retorne el objeto correcto
        mock_tokenizer_instance.__call__ = Mock(return_value=mock_encoded)
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_lora_training(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para entrenamiento con LoRA"""
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
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, self.config)
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar que se llamó get_peft_model
            mock_get_peft_model.assert_called_once()
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_molora_training(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para entrenamiento con MoLoRA"""
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
            
            # Verificar que se llamó get_peft_model
            mock_get_peft_model.assert_called_once()
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_ia3_training(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para entrenamiento con IA3"""
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
        
        trainer = PEFTUniversalTrainer(
            "test_model",
            self.model_path,
            self.output_dir,
            config
        )
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar que se llamó get_peft_model
            mock_get_peft_model.assert_called_once()
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_bitfit_training(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para entrenamiento con BitFit"""
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
        
        trainer = PEFTUniversalTrainer(
            "test_model",
            self.model_path,
            self.output_dir,
            config
        )
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar que se llamó get_peft_model
            mock_get_peft_model.assert_called_once()
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_prompt_tuning_training(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para entrenamiento con Prompt Tuning"""
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
        config = PromptTuningConfig(
            method=PEFTMethod.PROMPT_TUNING,
            num_virtual_tokens=20,
            prompt_tuning_init="random",
            target_modules=["embeddings"]
        )
        
        trainer = PEFTUniversalTrainer(
            "test_model",
            self.model_path,
            self.output_dir,
            config
        )
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar que se llamó get_peft_model
            mock_get_peft_model.assert_called_once()
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    @patch('LoRa_train.peft_universal_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.peft_universal_trainer.AutoTokenizer')
    @patch('LoRa_train.peft_universal_trainer.get_peft_model')
    def test_qlora_training(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para entrenamiento con QLoRA"""
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
        config = QLoRAConfig(
            method=PEFTMethod.QLORA,
            r=8,
            lora_alpha=16,
            bits=4,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        
        trainer = PEFTUniversalTrainer(
            "test_model",
            self.model_path,
            self.output_dir,
            config
        )
        
        # Asignar tokenizer mock al trainer
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, '_setup_training'), \
             patch.object(trainer, '_execute_training') as mock_execute, \
             patch.object(trainer, '_save_model'):
            
            mock_execute.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar que se llamó get_peft_model
            mock_get_peft_model.assert_called_once()
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    def test_invalid_method(self):
        """Test para método PEFT inválido"""
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
    
    def test_dataset_preparation(self):
        """Test para preparación de datasets"""
        # Configurar mocks básicos
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "gpt2"
        mock_model_instance.config.n_layer = 12
        
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        
        # Asignar tokenizer al trainer
        trainer = PEFTUniversalTrainer("test_model", self.model_path, self.output_dir, self.config)
        trainer.tokenizer = mock_tokenizer_instance
        
        # Test de preparación de datasets
        train_dataset, eval_dataset = trainer._prepare_datasets(self.test_data)
        
        # Verificar que se crearon los datasets
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(eval_dataset)


if __name__ == "__main__":
    unittest.main()
