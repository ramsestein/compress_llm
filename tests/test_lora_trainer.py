#!/usr/bin/env python3
"""
Tests para LoRATrainer
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

from LoRa_train.lora_trainer import LoRATrainer
from LoRa_train.lora_config import LoRAConfig, TrainingConfig, DataConfig


class TestLoRATrainer(unittest.TestCase):
    """Tests para LoRATrainer"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Crear archivos mock del modelo
        self._create_mock_model_files()
        
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
        mock_tokenizer_instance.__call__ = Mock(return_value=mock_encoded)
        mock_tokenizer_instance.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer_instance.encode_plus = Mock(return_value=mock_encoded)
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    @patch('LoRa_train.lora_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.lora_trainer.AutoTokenizer')
    @patch('LoRa_train.lora_trainer.get_peft_model')
    def test_lora_training_basic(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test básico de entrenamiento LoRA"""
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
        trainer = LoRATrainer("test_model", self.model_path, self.output_dir)
        
        # Mock del modelo y tokenizer cargados
        trainer.model = mock_model_instance
        trainer.tokenizer = mock_tokenizer_instance
        
        # Ejecutar entrenamiento
        with patch.object(trainer, 'train') as mock_train:
            mock_train.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            result = trainer.train(self.test_data)
            
            # Verificar resultado
            self.assertIn("train_loss", result)
            self.assertIn("eval_loss", result)
    
    @patch('LoRa_train.lora_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.lora_trainer.AutoTokenizer')
    @patch('LoRa_train.lora_trainer.get_peft_model')
    def test_target_modules_detection(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para detección de módulos objetivo"""
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
        trainer = LoRATrainer("test_model", self.model_path, self.output_dir)
        
        # Mock del modelo cargado
        trainer.model = mock_model_instance
        
        # Detectar módulos objetivo
        target_modules = trainer._detect_target_modules()
        
        # Verificar que se detectaron módulos
        self.assertIsInstance(target_modules, list)
        self.assertGreater(len(target_modules), 0)
    
    @patch('LoRa_train.lora_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.lora_trainer.AutoTokenizer')
    @patch('LoRa_train.lora_trainer.get_peft_model')
    def test_gpt2_architecture_detection(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para detección de arquitectura GPT-2"""
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
        trainer = LoRATrainer("test_model", self.model_path, self.output_dir)
        
        # Mock del modelo cargado
        trainer.model = mock_model_instance
        
        # Detectar módulos para GPT-2
        target_modules = trainer._detect_target_modules()
        
        # Verificar que se detectaron módulos específicos de GPT-2
        self.assertIsInstance(target_modules, list)
        self.assertGreater(len(target_modules), 0)
    
    @patch('LoRa_train.lora_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.lora_trainer.AutoTokenizer')
    @patch('LoRa_train.lora_trainer.get_peft_model')
    def test_llama_architecture_detection(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para detección de arquitectura Llama"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "llama"
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
        trainer = LoRATrainer("test_model", self.model_path, self.output_dir)
        
        # Mock del modelo cargado
        trainer.model = mock_model_instance
        
        # Detectar módulos para Llama
        target_modules = trainer._detect_target_modules()
        
        # Verificar que se detectaron módulos específicos de Llama
        self.assertIsInstance(target_modules, list)
        self.assertGreater(len(target_modules), 0)
    
    @patch('LoRa_train.lora_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.lora_trainer.AutoTokenizer')
    @patch('LoRa_train.lora_trainer.get_peft_model')
    def test_bert_architecture_detection(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para detección de arquitectura BERT"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "bert"
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
        trainer = LoRATrainer("test_model", self.model_path, self.output_dir)
        
        # Mock del modelo cargado
        trainer.model = mock_model_instance
        
        # Detectar módulos para BERT
        target_modules = trainer._detect_target_modules()
        
        # Verificar que se detectaron módulos específicos de BERT
        self.assertIsInstance(target_modules, list)
        self.assertGreater(len(target_modules), 0)
    
    @patch('LoRa_train.lora_trainer.AutoModelForCausalLM')
    @patch('LoRa_train.lora_trainer.AutoTokenizer')
    @patch('LoRa_train.lora_trainer.get_peft_model')
    def test_generic_module_detection(self, mock_get_peft_model, mock_tokenizer, mock_model):
        """Test para detección genérica de módulos"""
        # Configurar mocks
        mock_model_instance = Mock()
        mock_model_instance.config.model_type = "unknown"
        mock_model_instance.config.n_layer = 12
        mock_model_instance.state_dict.return_value = {"layer.0.weight": torch.randn(10, 10)}
        
        # Mock de named_modules para que retorne una lista iterable con módulos válidos
        mock_modules = [
            ("layer.0.linear", Mock()),
            ("layer.0.attention", Mock()),
            ("layer.1.linear", Mock()),
            ("layer.1.attention", Mock()),
            ("embeddings", Mock()),
            ("lm_head", Mock())
        ]
        mock_model_instance.named_modules.return_value = mock_modules
        
        mock_model.from_config.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        self._setup_mock_tokenizer(mock_tokenizer_instance)
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_peft_model
        
        # Crear trainer
        trainer = LoRATrainer("test_model", self.model_path, self.output_dir)
        
        # Mock del modelo cargado
        trainer.model = mock_model_instance
        
        # Detectar módulos genéricos
        generic_modules = trainer._generic_module_detection()
        
        # Verificar que se detectaron módulos genéricos
        self.assertIsInstance(generic_modules, list)
        self.assertGreater(len(generic_modules), 0)


if __name__ == "__main__":
    unittest.main()
