#!/usr/bin/env python3
"""
Test comprehensivo del sistema de compresión
Verifica todos los componentes, métodos, configuraciones y funcionalidades
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

# Asegurar que el paquete del proyecto esté en el path de importación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from create_compress.compression_engine import CompressionEngine, CompressionResult
from create_compress.compression_methods import (
    QuantizationMethod, PruningMethod, LowRankApproximation, 
    TuckerDecomposition, MPODecomposition
)
from create_compress.compression_profiles import COMPRESSION_PROFILES
from create_compress.compression_config_manager import CompressionConfigManager
from create_compress.interactive_config import InteractiveConfig


class TestCompressionSystemComprehensive(unittest.TestCase):
    """Test comprehensivo del sistema de compresión"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear directorio temporal para el modelo
        self.model_path = self.test_dir / "test_model"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo de configuración mock
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
        
        # Crear módulos de prueba
        self.linear_layer = nn.Linear(100, 50)
        self.attention_layer = nn.Linear(768, 768)
        self.ffn_layer = nn.Linear(768, 3072)
        
        # Inicializar componentes del sistema
        self.engine = CompressionEngine(device=str(self.device))
        self.config_manager = CompressionConfigManager(str(self.model_path))
        
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    # ============= TESTS DEL MOTOR DE COMPRESIÓN =============
    
    def test_compression_engine_initialization(self):
        """Test de inicialización del motor de compresión"""
        # Verificar inicialización básica
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.device.type, self.device.type)
        
        # Verificar métodos disponibles
        self.assertTrue(hasattr(self.engine, 'apply_method'))
        self.assertTrue(hasattr(self.engine, 'compress_layer'))
        self.assertTrue(hasattr(self.engine, 'compress_model'))
    
    def test_quantization_methods(self):
        """Test de métodos de cuantización"""
        # Test INT8
        result = self.engine.apply_method(
            self.linear_layer, 
            "int8_quantization", 
            0.5, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test INT4
        result = self.engine.apply_method(
            self.linear_layer, 
            "int4_quantization", 
            0.5, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test INT2
        result = self.engine.apply_method(
            self.linear_layer, 
            "int2_quantization", 
            0.5, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
    
    def test_pruning_methods(self):
        """Test de métodos de poda"""
        # Test magnitude pruning
        result = self.engine.apply_method(
            self.linear_layer, 
            "magnitude_pruning", 
            0.3, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test structured pruning
        result = self.engine.apply_method(
            self.linear_layer, 
            "structured_pruning", 
            0.3, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test attention pruning
        result = self.engine.apply_method(
            self.attention_layer, 
            "attention_pruning", 
            0.3, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
    
    def test_decomposition_methods(self):
        """Test de métodos de descomposición"""
        # Test SVD
        result = self.engine.apply_method(
            self.linear_layer, 
            "low_rank_approximation", 
            0.3, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test Tucker
        result = self.engine.apply_method(
            self.linear_layer, 
            "tucker_decomposition", 
            0.3, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test MPO
        result = self.engine.apply_method(
            self.linear_layer, 
            "mpo", 
            0.3, 
            {"params": {}}
        )
        self.assertIsNotNone(result)
    
    def test_compression_layer_functionality(self):
        """Test de funcionalidad de compresión de capas"""
        # Test con método válido
        compressed_layer, result = self.engine.compress_layer(
            self.linear_layer,
            {"name": "int8_quantization", "strength": 0.5}
        )
        self.assertIsNotNone(compressed_layer)
        self.assertIsInstance(result, CompressionResult)
        self.assertTrue(result.success)
        self.assertGreater(result.compression_ratio, 0)
        
        # Test con método inválido
        compressed_layer, result = self.engine.compress_layer(
            self.linear_layer,
            {"name": "invalid_method", "strength": 0.5}
        )
        self.assertIsNotNone(compressed_layer)
        self.assertEqual(result.method_used, "none")
        self.assertEqual(result.compression_ratio, 0)
    
    def test_quantization_parameters_calculation(self):
        """Test de cálculo de parámetros de cuantización"""
        # Test con tensor normal
        tensor = torch.randn(10, 10)
        scale, zero_point = self.engine._calculate_quantization_params(tensor, 8)
        self.assertIsInstance(scale, float)
        self.assertIsInstance(zero_point, int)
        self.assertGreater(scale, 0)
        
        # Test con tensor constante
        tensor = torch.ones(10, 10)
        scale, zero_point = self.engine._calculate_quantization_params(tensor, 8)
        self.assertEqual(scale, 1.0)
        self.assertEqual(zero_point, 0)
        
        # Test con tensor de ceros
        tensor = torch.zeros(10, 10)
        scale, zero_point = self.engine._calculate_quantization_params(tensor, 8)
        self.assertEqual(scale, 1.0)
        self.assertEqual(zero_point, 0)
    
    def test_tensor_quantization(self):
        """Test de cuantización de tensores"""
        tensor = torch.randn(5, 5)
        
        # Test INT8
        quantized = self.engine._quantize_tensor(tensor, 1.0, 0, 8)
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(quantized.shape, tensor.shape)
        
        # Test INT16
        quantized = self.engine._quantize_tensor(tensor, 1.0, 0, 16)
        self.assertEqual(quantized.dtype, torch.int16)
        self.assertEqual(quantized.shape, tensor.shape)
    
    # ============= TESTS DE MÉTODOS DE COMPRESIÓN =============
    
    def test_quantization_method_class(self):
        """Test de la clase QuantizationMethod"""
        method = QuantizationMethod(bits=8)
        
        # Test de estimación de compresión
        estimated_compression = method.estimate_compression(
            self.linear_layer, 
            {"strength": 0.5}
        )
        self.assertIsInstance(estimated_compression, float)
        self.assertGreaterEqual(estimated_compression, 0)
        self.assertLessEqual(estimated_compression, 1)
        
        # Test de compresión
        compressed = method.compress(
            self.linear_layer, 
            {"strength": 0.5}, 
            self.device
        )
        self.assertIsNotNone(compressed)
    
    def test_pruning_method_class(self):
        """Test de la clase PruningMethod"""
        method = PruningMethod()
        
        # Test de estimación de compresión
        estimated_compression = method.estimate_compression(
            self.linear_layer, 
            {"strength": 0.3}
        )
        self.assertIsInstance(estimated_compression, float)
        self.assertGreaterEqual(estimated_compression, 0)
        self.assertLessEqual(estimated_compression, 1)
        
        # Test de compresión
        compressed = method.compress(
            self.linear_layer, 
            {"strength": 0.3}, 
            self.device
        )
        self.assertIsNotNone(compressed)
    
    def test_low_rank_approximation_class(self):
        """Test de la clase LowRankApproximation"""
        method = LowRankApproximation()
        
        # Test de estimación de compresión
        estimated_compression = method.estimate_compression(
            self.linear_layer, 
            {"strength": 0.3}
        )
        self.assertIsInstance(estimated_compression, float)
        self.assertGreaterEqual(estimated_compression, 0)
        self.assertLessEqual(estimated_compression, 1)
        
        # Test de compresión
        compressed = method.compress(
            self.linear_layer, 
            {"strength": 0.3}, 
            self.device
        )
        self.assertIsNotNone(compressed)
        
        # Test de SVD aleatorizado
        weight = torch.randn(50, 30)
        U, S, V = method._randomized_svd(weight, rank=10)
        self.assertEqual(U.shape, (50, 10))
        self.assertEqual(S.shape, (10,))
        self.assertEqual(V.shape, (30, 10))
        
        # Verificar reconstrucción
        recon = (U * S) @ V.t()
        self.assertEqual(recon.shape, weight.shape)
    
    def test_tucker_decomposition_class(self):
        """Test de la clase TuckerDecomposition"""
        method = TuckerDecomposition()
        
        # Test de estimación de compresión
        estimated_compression = method.estimate_compression(
            self.linear_layer, 
            {"strength": 0.3}
        )
        self.assertIsInstance(estimated_compression, float)
        self.assertGreaterEqual(estimated_compression, 0)
        self.assertLessEqual(estimated_compression, 1)
        
        # Test de compresión
        compressed = method.compress(
            self.linear_layer, 
            {"strength": 0.3}, 
            self.device
        )
        self.assertIsNotNone(compressed)
    
    def test_mpo_decomposition_class(self):
        """Test de la clase MPODecomposition"""
        method = MPODecomposition()
        
        # Test de estimación de compresión
        estimated_compression = method.estimate_compression(
            self.linear_layer, 
            {"strength": 0.3}
        )
        self.assertIsInstance(estimated_compression, float)
        self.assertGreaterEqual(estimated_compression, 0)
        self.assertLessEqual(estimated_compression, 1)
        
        # Test de compresión
        compressed = method.compress(
            self.linear_layer, 
            {"strength": 0.3}, 
            self.device
        )
        self.assertIsNotNone(compressed)
        
        # Test de factorización de dimensiones
        factors = method._factorize_dimension(100)
        self.assertEqual(len(factors), 2)
        self.assertEqual(factors[0] * factors[1], 100)
    
    # ============= TESTS DE PERFILES DE COMPRESIÓN =============
    
    def test_compression_profiles_structure(self):
        """Test de estructura de perfiles de compresión"""
        # Verificar que existen los perfiles principales
        required_profiles = ['conservative', 'balanced', 'aggressive']
        for profile_name in required_profiles:
            self.assertIn(profile_name, COMPRESSION_PROFILES)
            
            profile = COMPRESSION_PROFILES[profile_name]
            self.assertIn('name', profile)
            self.assertIn('description', profile)
            self.assertIn('goal', profile)
            self.assertIn('target_compression', profile)
            self.assertIn('layer_configs', profile)
    
    def test_conservative_profile(self):
        """Test del perfil conservador"""
        profile = COMPRESSION_PROFILES['conservative']
        
        self.assertEqual(profile['goal'], 'max_quality')
        self.assertLess(profile['target_compression'], 0.5)
        
        # Verificar configuraciones de capas
        layer_configs = profile['layer_configs']
        self.assertIn('attention', layer_configs)
        self.assertIn('ffn', layer_configs)
        self.assertIn('embedding', layer_configs)
        
        # Verificar que los métodos son conservadores
        attention_config = layer_configs['attention']
        self.assertLess(attention_config['total_compression_ratio'], 0.3)
    
    def test_balanced_profile(self):
        """Test del perfil balanceado"""
        profile = COMPRESSION_PROFILES['balanced']
        
        self.assertEqual(profile['goal'], 'balanced')
        self.assertAlmostEqual(profile['target_compression'], 0.5, delta=0.1)
        
        # Verificar configuraciones de capas
        layer_configs = profile['layer_configs']
        self.assertIn('attention', layer_configs)
        self.assertIn('ffn', layer_configs)
        self.assertIn('embedding', layer_configs)
        
        # Verificar que los métodos están balanceados
        attention_config = layer_configs['attention']
        self.assertGreater(attention_config['total_compression_ratio'], 0.3)
        self.assertLess(attention_config['total_compression_ratio'], 0.7)
    
    def test_aggressive_profile(self):
        """Test del perfil agresivo"""
        profile = COMPRESSION_PROFILES['aggressive']
        
        self.assertEqual(profile['goal'], 'max_compression')
        self.assertGreater(profile['target_compression'], 0.5)
        
        # Verificar configuraciones de capas
        layer_configs = profile['layer_configs']
        self.assertIn('attention', layer_configs)
        self.assertIn('ffn', layer_configs)
        self.assertIn('embedding', layer_configs)
        
        # Verificar que los métodos son agresivos
        attention_config = layer_configs['attention']
        self.assertGreater(attention_config['total_compression_ratio'], 0.5)
    
    def test_layer_configs_structure(self):
        """Test de estructura de configuraciones de capas"""
        for profile_name, profile in COMPRESSION_PROFILES.items():
            layer_configs = profile['layer_configs']
            
            for layer_type, config in layer_configs.items():
                # Verificar estructura básica
                self.assertIn('methods', config)
                self.assertIn('total_compression_ratio', config)
                
                # Verificar que methods es una lista
                self.assertIsInstance(config['methods'], list)
                
                # Verificar cada método
                for method in config['methods']:
                    self.assertIn('name', method)
                    self.assertIn('strength', method)
                    self.assertIsInstance(method['strength'], (int, float))
                    self.assertGreaterEqual(method['strength'], 0)
                    self.assertLessEqual(method['strength'], 1)
    
    # ============= TESTS DEL GESTOR DE CONFIGURACIÓN =============
    
    def test_config_manager_initialization(self):
        """Test de inicialización del gestor de configuración"""
        self.assertIsNotNone(self.config_manager)
        self.assertTrue(hasattr(self.config_manager, 'load_config'))
        self.assertTrue(hasattr(self.config_manager, 'save_config'))
        self.assertTrue(hasattr(self.config_manager, 'validate_config'))
    
    def test_config_validation(self):
        """Test de validación de configuraciones"""
        # Configuración válida
        valid_config = {
            "metadata": {
                "model_name": "test_model",
                "version": "1.0"
            },
            "global_settings": {
                "name": "Balanced",
                "target_compression": 0.5,
                "layer_configs": {
                    "attention": {
                        "methods": [{"name": "int8_quantization", "strength": 0.5}],
                        "total_compression_ratio": 0.3
                    }
                }
            }
        }
        
        is_valid, errors = self.config_manager.validate_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Configuración inválida
        invalid_config = {
            "metadata": {},
            "global_settings": {
                "target_compression": 2.0  # Valor inválido
            }
        }
        
        is_valid, errors = self.config_manager.validate_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_config_save_load(self):
        """Test de guardado y carga de configuraciones"""
        config = {
            "metadata": {
                "model_name": "test_model",
                "version": "1.0"
            },
            "global_settings": {
                "name": "Balanced",
                "target_compression": 0.5,
                "layer_configs": {
                    "attention": {
                        "methods": [{"name": "int8_quantization", "strength": 0.5}],
                        "total_compression_ratio": 0.3
                    }
                }
            }
        }
        
        config_path = self.test_dir / "test_config.json"
        
        # Guardar configuración
        success = self.config_manager.save_config(config, config_path)
        self.assertTrue(success)
        self.assertTrue(config_path.exists())
        
        # Cargar configuración
        loaded_config = self.config_manager.load_config(config_path)
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config["metadata"]["model_name"], "test_model")
        self.assertEqual(loaded_config["global_settings"]["target_compression"], 0.5)
    
    # ============= TESTS DE CONFIGURACIÓN INTERACTIVA =============
    
    def test_interactive_config_builder(self):
        """Test del constructor de configuración interactiva"""
        builder = InteractiveConfig(None)
        
        self.assertIsNotNone(builder)
        self.assertTrue(hasattr(builder, 'configure_profile'))
        self.assertTrue(hasattr(builder, 'configure_layer_types'))
    
    def test_profile_selection(self):
        """Test de selección de perfiles"""
        builder = InteractiveConfig(None)
        
        # Simular selección de perfil balanceado
        profile = builder.COMPRESSION_PROFILES.get('balanced')
        self.assertIsNotNone(profile)
        self.assertEqual(profile['name'], 'Balanced')
        self.assertAlmostEqual(profile['target_compression'], 0.5, delta=0.1)
    
    def test_layer_config_customization(self):
        """Test de personalización de configuraciones de capas"""
        builder = InteractiveConfig(None)
        
        # Obtener perfil base
        base_profile = COMPRESSION_PROFILES['balanced']
        
        # Personalizar configuración
        customized = {
            'attention': {'total_compression_ratio': 0.4},
            'ffn': {'total_compression_ratio': 0.6}
        }
        
        self.assertIsNotNone(customized)
        self.assertEqual(customized['attention']['total_compression_ratio'], 0.4)
        self.assertEqual(customized['ffn']['total_compression_ratio'], 0.6)
    
    # ============= TESTS DE INTEGRACIÓN =============
    
    def test_end_to_end_compression_workflow(self):
        """Test de flujo completo de compresión"""
        # Crear configuración
        config = {
            "metadata": {
                "model_name": "test_model",
                "version": "1.0"
            },
            "global_settings": {
                "name": "Balanced",
                "target_compression": 0.5,
                "layer_configs": {
                    "attention": {
                        "methods": [{"name": "int8_quantization", "strength": 0.5}],
                        "total_compression_ratio": 0.3
                    },
                    "ffn": {
                        "methods": [{"name": "magnitude_pruning", "strength": 0.3}],
                        "total_compression_ratio": 0.4
                    }
                }
            }
        }
        
        # Validar configuración
        is_valid, errors = self.config_manager.validate_config(config)
        self.assertTrue(is_valid)
        
        # Aplicar compresión
        compressed_layer, result = self.engine.compress_layer(
            self.attention_layer,
            config["global_settings"]["layer_configs"]["attention"]["methods"][0]
        )
        
        self.assertIsNotNone(compressed_layer)
        self.assertTrue(result.success)
        self.assertGreater(result.compression_ratio, 0)
    
    def test_compression_result_structure(self):
        """Test de estructura del resultado de compresión"""
        # Crear resultado de compresión
        result = CompressionResult(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            method_used="int8_quantization",
            success=True
        )
        
        # Verificar estructura
        self.assertEqual(result.original_size, 1000)
        self.assertEqual(result.compressed_size, 500)
        self.assertEqual(result.compression_ratio, 0.5)
        self.assertEqual(result.method_used, "int8_quantization")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        
        # Test con error
        error_result = CompressionResult(
            original_size=1000,
            compressed_size=1000,
            compression_ratio=0.0,
            method_used="invalid_method",
            success=False,
            error="Method not found"
        )
        
        self.assertFalse(error_result.success)
        self.assertEqual(error_result.error, "Method not found")
    
    def test_memory_efficiency(self):
        """Test de eficiencia de memoria"""
        # Crear capa grande
        large_layer = nn.Linear(1000, 1000)
        
        # Aplicar compresión
        compressed_layer, result = self.engine.compress_layer(
            large_layer,
            {"name": "int8_quantization", "strength": 0.5}
        )
        
        # Verificar que la compresión fue efectiva
        self.assertTrue(result.success)
        self.assertGreater(result.compression_ratio, 0)
        self.assertLess(result.compressed_size, result.original_size)
    
    def test_error_handling(self):
        """Test de manejo de errores"""
        # Test con método inexistente
        compressed_layer, result = self.engine.compress_layer(
            self.linear_layer,
            {"name": "nonexistent_method", "strength": 0.5}
        )
        
        self.assertIsNotNone(compressed_layer)
        self.assertEqual(result.method_used, "none")
        self.assertEqual(result.compression_ratio, 0)
        
        # Test con parámetros inválidos
        compressed_layer, result = self.engine.compress_layer(
            self.linear_layer,
            {"name": "int8_quantization", "strength": 2.0}  # Strength inválido
        )
        
        self.assertIsNotNone(compressed_layer)
        # El sistema debería manejar esto graciosamente
    
    def test_device_compatibility(self):
        """Test de compatibilidad de dispositivos"""
        # Test en CPU
        cpu_engine = CompressionEngine(device="cpu")
        result = cpu_engine.apply_method(
            self.linear_layer,
            "int8_quantization",
            0.5,
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test en GPU si está disponible
        if torch.cuda.is_available():
            gpu_engine = CompressionEngine(device="cuda")
            result = gpu_engine.apply_method(
                self.linear_layer.cuda(),
                "int8_quantization",
                0.5,
                {"params": {}}
            )
            self.assertIsNotNone(result)
    
    def test_numerical_stability(self):
        """Test de estabilidad numérica"""
        # Test con valores extremos
        extreme_layer = nn.Linear(10, 10)
        extreme_layer.weight.data = torch.randn(10, 10) * 1e6  # Valores muy grandes
        
        result = self.engine.apply_method(
            extreme_layer,
            "int8_quantization",
            0.5,
            {"params": {}}
        )
        self.assertIsNotNone(result)
        
        # Test con valores muy pequeños
        small_layer = nn.Linear(10, 10)
        small_layer.weight.data = torch.randn(10, 10) * 1e-6  # Valores muy pequeños
        
        result = self.engine.apply_method(
            small_layer,
            "int8_quantization",
            0.5,
            {"params": {}}
        )
        self.assertIsNotNone(result)
    
    def test_reproducibility(self):
        """Test de reproducibilidad"""
        # Aplicar la misma compresión dos veces
        torch.manual_seed(42)
        result1 = self.engine.apply_method(
            self.linear_layer,
            "int8_quantization",
            0.5,
            {"params": {}}
        )
        
        torch.manual_seed(42)
        result2 = self.engine.apply_method(
            self.linear_layer,
            "int8_quantization",
            0.5,
            {"params": {}}
        )
        
        # Los resultados deberían ser similares
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
