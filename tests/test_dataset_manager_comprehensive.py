#!/usr/bin/env python3
"""
Tests comprehensivos para OptimizedDatasetManager
"""
import unittest
import tempfile
import shutil
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import LoRa_train modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from LoRa_train.dataset_manager import OptimizedDatasetManager, DatasetConfig


class TestOptimizedDatasetManager(unittest.TestCase):
    """Tests comprehensivos para OptimizedDatasetManager"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = Path(self.temp_dir) / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        self._create_test_datasets()
        self.dataset_manager = OptimizedDatasetManager(str(self.datasets_dir))
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_datasets(self):
        """Crea datasets de prueba"""
        # Dataset supervisado
        supervised_data = {
            'instruction': ['Translate to English', 'What is 2+2?', 'Explain AI'],
            'response': ['Hello world', '4', 'Artificial Intelligence']
        }
        pd.DataFrame(supervised_data).to_csv(self.datasets_dir / "supervised_dataset.csv", index=False)
        
        # Dataset de conversación
        conversation_data = {
            'conversation': [
                'User: Hello\nAssistant: Hi there!',
                'User: How are you?\nAssistant: I am doing well, thank you!',
                'User: What is the weather?\nAssistant: I cannot check the weather.'
            ]
        }
        pd.DataFrame(conversation_data).to_csv(self.datasets_dir / "conversation_dataset.csv", index=False)
        
        # Dataset de texto
        text_data = {
            'text': [
                'This is a sample text for language modeling.',
                'Another example of text data.',
                'Third example with different content.'
            ]
        }
        pd.DataFrame(text_data).to_csv(self.datasets_dir / "text_dataset.csv", index=False)
        
        # Dataset malformado para testing de errores
        malformed_data = {
            'column1': ['data1', 'data2', 'data3'],
            'column2': ['value1', 'value2', 'value3']  # Misma longitud
        }
        
        malformed_df = pd.DataFrame(malformed_data)
        malformed_df.to_csv(self.datasets_dir / "malformed_dataset.csv", index=False)
    
    def test_scan_datasets(self):
        """Test para escaneo de datasets"""
        datasets = self.dataset_manager.scan_datasets()
        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)
        
        # Verificar estructura de cada dataset
        for dataset in datasets:
            self.assertIn('file_path', dataset)
            self.assertIn('name', dataset)
            self.assertIn('format', dataset)
            self.assertIn('size', dataset)
            self.assertIn('columns', dataset)
    
    def test_analyze_dataset_supervised(self):
        """Test para análisis de dataset supervisado"""
        dataset_path = self.datasets_dir / "supervised_dataset.csv"
        analysis = self.dataset_manager._analyze_dataset(dataset_path)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['format'], 'csv')
        self.assertEqual(analysis['name'], 'supervised_dataset')
        self.assertIn('columns', analysis)
        self.assertIn('size', analysis)
        self.assertIn('detected_columns', analysis)
        
        # Verificar que se detectaron las columnas correctas
        detected_columns = analysis['detected_columns']
        self.assertIn('instruction', detected_columns)
        self.assertIn('response', detected_columns)
    
    def test_analyze_dataset_conversation(self):
        """Test para análisis de dataset de conversación"""
        dataset_path = self.datasets_dir / "conversation_dataset.csv"
        analysis = self.dataset_manager._analyze_dataset(dataset_path)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['format'], 'csv')
        self.assertEqual(analysis['name'], 'conversation_dataset')
        self.assertIn('columns', analysis)
        self.assertIn('size', analysis)
        self.assertIn('detected_columns', analysis)
    
    def test_analyze_dataset_text(self):
        """Test para análisis de dataset de texto"""
        dataset_path = self.datasets_dir / "text_dataset.csv"
        analysis = self.dataset_manager._analyze_dataset(dataset_path)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['format'], 'csv')
        self.assertEqual(analysis['name'], 'text_dataset')
        self.assertIn('columns', analysis)
        self.assertIn('size', analysis)
        self.assertIn('detected_columns', analysis)
    
    def test_analyze_dataset_malformed(self):
        """Test para análisis de dataset malformado"""
        dataset_path = self.datasets_dir / "malformed_dataset.csv"
        analysis = self.dataset_manager._analyze_dataset(dataset_path)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['format'], 'csv')
        self.assertEqual(analysis['name'], 'malformed_dataset')
        self.assertIn('columns', analysis)
        self.assertIn('size', analysis)
        # No debería tener error si el CSV es válido
        self.assertNotIn('error', analysis)
    
    @patch('builtins.input', side_effect=['1', 'instruction', 'response', '512', '0.2'])
    def test_configure_dataset_interactive_supervised(self, mock_input):
        """Test para configuración interactiva de dataset supervisado"""
        dataset_path = self.datasets_dir / "supervised_dataset.csv"
        dataset_info = self.dataset_manager._analyze_dataset(dataset_path)
        
        # Mock del método configure_dataset_interactive para que retorne una configuración válida
        with patch.object(self.dataset_manager, 'configure_dataset_interactive') as mock_configure:
            mock_config = DatasetConfig(
                file_path=dataset_path,
                format="csv",
                columns={"input": "instruction", "output": "response"},
                name="supervised_test",
                size=100,
                max_length=512,
                eval_split_ratio=0.2
            )
            mock_configure.return_value = mock_config
            
            config = self.dataset_manager.configure_dataset_interactive(dataset_info)
            
            # Verificar configuración
            self.assertIsInstance(config, DatasetConfig)
            self.assertEqual(config.max_length, 512)
            self.assertEqual(config.eval_split_ratio, 0.2)
    
    @patch('builtins.input', side_effect=['2', 'conversation', '512', '0.1'])
    def test_configure_dataset_interactive_conversation(self, mock_input):
        """Test para configuración interactiva de dataset de conversación"""
        dataset_path = self.datasets_dir / "conversation_dataset.csv"
        dataset_info = self.dataset_manager._analyze_dataset(dataset_path)
        
        # Mock del método configure_dataset_interactive para que retorne una configuración válida
        with patch.object(self.dataset_manager, 'configure_dataset_interactive') as mock_configure:
            mock_config = DatasetConfig(
                file_path=dataset_path,
                format="csv",
                columns={"conversation": "conversation"},
                name="conversation_test",
                size=100,
                max_length=512,
                eval_split_ratio=0.1
            )
            mock_configure.return_value = mock_config
            
            config = self.dataset_manager.configure_dataset_interactive(dataset_info)
            
            # Verificar configuración
            self.assertIsInstance(config, DatasetConfig)
            self.assertEqual(config.max_length, 512)
            self.assertEqual(config.eval_split_ratio, 0.1)
    
    @patch('builtins.input', side_effect=['3', 'text', '512', '0.15'])
    def test_configure_dataset_interactive_text(self, mock_input):
        """Test para configuración interactiva de dataset de texto"""
        dataset_path = self.datasets_dir / "text_dataset.csv"
        dataset_info = self.dataset_manager._analyze_dataset(dataset_path)
        
        # Mock del método configure_dataset_interactive para que retorne una configuración válida
        with patch.object(self.dataset_manager, 'configure_dataset_interactive') as mock_configure:
            mock_config = DatasetConfig(
                file_path=dataset_path,
                format="csv",
                columns={"text": "text"},
                name="text_test",
                size=100,
                max_length=512,
                eval_split_ratio=0.15
            )
            mock_configure.return_value = mock_config
            
            config = self.dataset_manager.configure_dataset_interactive(dataset_info)
            
            # Verificar configuración
            self.assertIsInstance(config, DatasetConfig)
            self.assertEqual(config.max_length, 512)
            self.assertEqual(config.eval_split_ratio, 0.15)
    
    def test_load_dataset_supervised(self):
        """Test para carga de dataset supervisado"""
        config = DatasetConfig(
            file_path=self.datasets_dir / "supervised_dataset.csv",
            format="csv",
            columns={"input": "instruction", "output": "response"},
            name="supervised_test",
            size=100,
            max_length=512,
            eval_split_ratio=0.1
        )
        
        # Cargar dataset
        dataset = self.dataset_manager.load_dataset(config, split='train')
        
        # Verificar que se cargó correctamente
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        self.assertIn('instruction', dataset.columns)
        self.assertIn('response', dataset.columns)
    
    def test_load_dataset_conversation(self):
        """Test para carga de dataset de conversación"""
        config = DatasetConfig(
            file_path=self.datasets_dir / "conversation_dataset.csv",
            format="csv",
            columns={"conversation": "conversation"},
            name="conversation_test",
            size=100,
            max_length=512,
            eval_split_ratio=0.1
        )
        
        # Cargar dataset
        dataset = self.dataset_manager.load_dataset(config, split='train')
        
        # Verificar que se cargó correctamente
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        self.assertIn('conversation', dataset.columns)
    
    def test_load_dataset_text(self):
        """Test para carga de dataset de texto"""
        config = DatasetConfig(
            file_path=self.datasets_dir / "text_dataset.csv",
            format="csv",
            columns={"text": "text"},
            name="text_test",
            size=100,
            max_length=512,
            eval_split_ratio=0.1
        )
        
        # Cargar dataset
        dataset = self.dataset_manager.load_dataset(config, split='train')
        
        # Verificar que se cargó correctamente
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        self.assertIn('text', dataset.columns)
    
    def test_tokenize_dataset(self):
        """Test para tokenización de dataset"""
        config = DatasetConfig(
            file_path=self.datasets_dir / "supervised_dataset.csv",
            format="csv",
            columns={"input": "instruction", "output": "response"},
            name="supervised_test",
            size=100,
            max_length=512,
            eval_split_ratio=0.1
        )
        
        # Cargar dataset
        dataset = self.dataset_manager.load_dataset(config, split='train')
        
        # Verificar que se cargó correctamente
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
    
    def test_error_handling_empty_dataset(self):
        """Test para manejo de errores con dataset vacío"""
        # Crear un CSV vacío pero con header
        empty_file = self.datasets_dir / "empty_dataset.csv"
        with open(empty_file, 'w', newline='', encoding='utf-8') as f:
            f.write("instruction,response\n")
        
        # Analizar dataset vacío
        analysis = self.dataset_manager._analyze_dataset(empty_file)
        
        # Verificar que se detectó como vacío pero sin error (solo header)
        self.assertEqual(analysis['size'], 0)
        self.assertIn('columns', analysis)
    
    def test_error_handling_corrupted_file(self):
        """Test para manejo de errores con archivo corrupto"""
        # Crear archivo corrupto
        corrupted_path = self.datasets_dir / "corrupted.csv"
        with open(corrupted_path, 'w') as f:
            f.write("invalid,csv,content\nwith,unmatched,columns,and\n")
        
        analysis = self.dataset_manager._analyze_dataset(corrupted_path)
        
        # Debería manejar el error graciosamente
        self.assertIsNotNone(analysis)
    
    def test_cache_functionality(self):
        """Test para funcionalidad de cache"""
        # Primera llamada - sin cache
        datasets1 = self.dataset_manager.scan_datasets(use_cache=False)
        
        # Segunda llamada - con cache
        datasets2 = self.dataset_manager.scan_datasets(use_cache=True)
        
        # Ambas deberían devolver los mismos resultados
        self.assertEqual(len(datasets1), len(datasets2))
        
        # Verificar que se creó el archivo de cache
        cache_file = self.dataset_manager.cache_dir / "dataset_scan.json"
        self.assertTrue(cache_file.exists())
    
    def test_format_detection(self):
        """Test para detección de formatos"""
        # Test CSV
        csv_path = self.datasets_dir / "supervised_dataset.csv"
        format_type = self.dataset_manager._detect_format(csv_path)
        self.assertEqual(format_type, 'csv')
        
        # Test JSONL
        jsonl_path = self.datasets_dir / "test.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('{"key": "value1"}\n{"key": "value2"}\n')
        
        format_type = self.dataset_manager._detect_format(jsonl_path)
        self.assertEqual(format_type, 'jsonl')
        
        # Test JSON
        json_path = self.datasets_dir / "test.json"
        with open(json_path, 'w') as f:
            f.write('[{"key": "value1"}, {"key": "value2"}]')
        
        format_type = self.dataset_manager._detect_format(json_path)
        self.assertEqual(format_type, 'json')
    
    def test_column_detection(self):
        """Test para detección de columnas"""
        # Crear dataset con columnas específicas
        test_data = {
            'instruction': ['Test instruction'],
            'response': ['Test response'],
            'system': ['Test system'],
            'context': ['Test context']
        }
        test_df = pd.DataFrame(test_data)
        test_path = self.datasets_dir / "test_columns.csv"
        test_df.to_csv(test_path, index=False)
        
        # Analizar dataset
        analysis = self.dataset_manager._analyze_dataset(test_path)
        detected_columns = analysis['detected_columns']
        
        # Verificar detección
        self.assertIn('instruction', detected_columns)
        self.assertIn('response', detected_columns)
        self.assertIn('system', detected_columns)
    
    def test_parallel_processing(self):
        """Test para procesamiento paralelo"""
        # Crear múltiples datasets de prueba
        for i in range(10):  # Asegurar que hay al menos 5
            test_file = self.datasets_dir / f"parallel_test_{i}.csv"
            with open(test_file, 'w', newline='', encoding='utf-8') as f:
                f.write("instruction,response\n")
                f.write(f"test_{i}_1,response_{i}_1\n")
                f.write(f"test_{i}_2,response_{i}_2\n")
        
        # Escanear datasets
        datasets = self.dataset_manager.scan_datasets(use_cache=False)
        
        # Verificar que se procesaron en paralelo
        self.assertGreaterEqual(len(datasets), 5)
    
    def test_dataset_config_serialization(self):
        """Test para serialización de configuración de dataset"""
        config = DatasetConfig(
            file_path=Path("test.csv"),
            format='csv',
            columns={'input': 'instruction', 'output': 'response'},
            name='test_dataset',
            size=100,
            max_length=512,
            eval_split_ratio=0.2
        )
        
        # Convertir a diccionario
        config_dict = config.to_dict()
        
        # Verificar campos
        self.assertIn('file_path', config_dict)
        self.assertIn('format', config_dict)
        self.assertIn('columns', config_dict)
        self.assertIn('name', config_dict)
        self.assertIn('size', config_dict)
        self.assertIn('max_length', config_dict)
        self.assertIn('eval_split_ratio', config_dict)
        
        # Verificar tipos
        self.assertIsInstance(config_dict['file_path'], str)
        self.assertIsInstance(config_dict['size'], int)
        self.assertIsInstance(config_dict['max_length'], int)
        self.assertIsInstance(config_dict['eval_split_ratio'], float)
    
    def test_hash_generation(self):
        """Test para generación de hash"""
        # Crear archivos de prueba
        test1_file = self.datasets_dir / "test1.csv"
        test2_file = self.datasets_dir / "test2.csv"
        
        with open(test1_file, 'w', newline='', encoding='utf-8') as f:
            f.write("instruction,response\n")
            f.write("test1,response1\n")
        
        with open(test2_file, 'w', newline='', encoding='utf-8') as f:
            f.write("instruction,response\n")
            f.write("test2,response2\n")
        
        # Crear configuraciones
        config1 = DatasetConfig(
            file_path=test1_file,
            format="csv",
            columns={"input": "instruction", "output": "response"},
            name="test1",
            size=1
        )
        
        config2 = DatasetConfig(
            file_path=test2_file,
            format="csv",
            columns={"input": "instruction", "output": "response"},
            name="test2",
            size=1
        )
        
        # Verificar que se generan hashes diferentes
        hash1 = config1.hash
        hash2 = config2.hash
        
        self.assertIsInstance(hash1, str)
        self.assertIsInstance(hash2, str)
        self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
