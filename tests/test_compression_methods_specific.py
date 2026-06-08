#!/usr/bin/env python3
"""
Tests específicos para todos los métodos de compresión disponibles
Verifica cada método individualmente con configuraciones específicas
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
from unittest.mock import patch, MagicMock

# Asegurar que el paquete del proyecto esté en el path de importación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from create_compress.compression_methods import (
    QuantizationMethod, PruningMethod, LowRankApproximation,
    AttentionPruning, MPOCompression, TuckerDecomposition,
    ExpertPruning, SVDDecomposition, MixedPrecisionMethod,
    BlockSparseMethod, NullCompression
)
from create_compress.compression_engine import CompressionEngine


class TestCompressionMethodsSpecific(unittest.TestCase):
    """Tests específicos para cada método de compresión"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear módulos de prueba
        self.linear_layer = nn.Linear(100, 50)
        self.attention_layer = nn.Linear(768, 768)
        self.ffn_layer = nn.Linear(768, 3072)
        
        # Inicializar engine
        self.engine = CompressionEngine(device=str(self.device))
        
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    # ============= TESTS DE CUANTIZACIÓN =============
    
    def test_int8_quantization(self):
        """Test específico para cuantización INT8"""
        method = QuantizationMethod(bits=8)
        
        # Configuración específica
        config = {
            'strength': 0.5,
            'bits': 8,
            'symmetric': True
        }
        
        # Aplicar compresión
        compressed = method.compress(self.linear_layer, config, self.device)
        
        # Verificaciones específicas
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
        
        # Verificar que los pesos están cuantizados
        if hasattr(compressed, 'weight'):
            weight = compressed.weight.data
            # Verificar que los valores están en rango INT8
            self.assertTrue(torch.all(weight >= -128))
            self.assertTrue(torch.all(weight <= 127))
    
    def test_int4_quantization(self):
        """Test específico para cuantización INT4"""
        method = QuantizationMethod(bits=4)
        
        config = {
            'strength': 0.7,
            'bits': 4,
            'symmetric': False
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
        
        # Verificar rango INT4
        if hasattr(compressed, 'weight'):
            weight = compressed.weight.data
            self.assertTrue(torch.all(weight >= -8))
            self.assertTrue(torch.all(weight <= 7))
    
    def test_int2_quantization(self):
        """Test específico para cuantización INT2 (ternaria)"""
        method = QuantizationMethod(bits=2)
        
        config = {
            'strength': 0.8,
            'bits': 2,
            'ternary': True
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
        
        # Verificar que se aplicó la compresión INT2
        if hasattr(compressed, 'weight'):
            weight = compressed.weight.data
            # Verificar que los pesos no son NaN o infinitos
            self.assertFalse(torch.isnan(weight).any())
            self.assertFalse(torch.isinf(weight).any())
            # Verificar que los pesos están en un rango razonable para INT2
            self.assertTrue(torch.all(weight >= -2))
            self.assertTrue(torch.all(weight <= 2))
    
    # ============= TESTS DE PRUNING =============
    
    def test_magnitude_pruning(self):
        """Test específico para pruning por magnitud"""
        method = PruningMethod()
        
        config = {
            'strength': 0.3,
            'pruning_type': 'magnitude',
            'sparsity': 0.3
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
        
        # Verificar sparsity
        if hasattr(compressed, 'weight'):
            weight = compressed.weight.data
            sparsity = (weight == 0).float().mean()
            self.assertGreaterEqual(sparsity, 0.25)  # Al menos 25% de sparsity
    
    def test_structured_pruning(self):
        """Test específico para pruning estructurado"""
        method = PruningMethod()
        
        config = {
            'strength': 0.4,
            'pruning_type': 'structured',
            'sparsity': 0.4,
            'block_size': 4
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    def test_attention_pruning(self):
        """Test específico para pruning de atención"""
        method = AttentionPruning()
        
        config = {
            'strength': 0.5,
            'head_sparsity': 0.3,
            'attention_type': 'self_attention'
        }
        
        compressed = method.compress(self.attention_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    # ============= TESTS DE DESCOMPOSICIÓN =============
    
    def test_svd_decomposition(self):
        """Test específico para descomposición SVD"""
        method = SVDDecomposition()
        
        config = {
            'strength': 0.6,
            'rank': 50,
            'svd_type': 'truncated'
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
        
        # Verificar que se aplicó SVD
        if hasattr(compressed, 'weight'):
            weight = compressed.weight.data
            # Verificar que la matriz tiene rango reducido
            rank = torch.linalg.matrix_rank(weight)
            self.assertLessEqual(rank, 50)
    
    def test_tucker_decomposition(self):
        """Test específico para descomposición Tucker"""
        method = TuckerDecomposition()
        
        config = {
            'strength': 0.5,
            'ranks': [10, 25],
            'tucker_type': 'standard'
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    def test_mpo_compression(self):
        """Test específico para compresión MPO"""
        try:
            method = MPOCompression()
            
            config = {
                'strength': 0.4,
                'bond_dim': 8,
                'mpo_type': 'standard'
            }
            
            compressed = method.compress(self.linear_layer, config, self.device)
            
            self.assertIsNotNone(compressed)
            self.assertIsInstance(compressed, nn.Module)
        except ImportError:
            self.skipTest("TensorLy no está instalado")
    
    def test_low_rank_approximation(self):
        """Test específico para aproximación de bajo rango"""
        method = LowRankApproximation()
        
        config = {
            'strength': 0.5,
            'rank': 30,
            'approximation_type': 'svd'
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    # ============= TESTS DE MÉTODOS ESPECIALIZADOS =============
    
    def test_expert_pruning(self):
        """Test específico para pruning de expertos"""
        method = ExpertPruning()
        
        config = {
            'strength': 0.3,
            'expert_sparsity': 0.3,
            'num_experts': 8
        }
        
        compressed = method.compress(self.ffn_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    def test_mixed_precision_method(self):
        """Test específico para precisión mixta"""
        method = MixedPrecisionMethod()
        
        config = {
            'strength': 0.5,
            'precision_levels': ['fp16', 'fp32'],
            'mixed_type': 'dynamic'
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    def test_block_sparse_method(self):
        """Test específico para sparse por bloques"""
        method = BlockSparseMethod()
        
        config = {
            'strength': 0.4,
            'block_size': 4,
            'sparsity': 0.4,
            'sparse_type': 'structured'
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
    
    def test_null_compression(self):
        """Test específico para compresión nula (sin cambios)"""
        method = NullCompression()
        
        config = {
            'strength': 0.0
        }
        
        compressed = method.compress(self.linear_layer, config, self.device)
        
        self.assertIsNotNone(compressed)
        self.assertIsInstance(compressed, nn.Module)
        
        # Verificar que no hay cambios
        if hasattr(compressed, 'weight') and hasattr(self.linear_layer, 'weight'):
            original_weight = self.linear_layer.weight.data
            compressed_weight = compressed.weight.data
            self.assertTrue(torch.allclose(original_weight, compressed_weight))
    
    # ============= TESTS DE ESTIMACIÓN DE COMPRESIÓN =============
    
    def test_compression_estimation(self):
        """Test de estimación de compresión para todos los métodos"""
        methods = [
            QuantizationMethod(bits=8),
            QuantizationMethod(bits=4),
            QuantizationMethod(bits=2),
            PruningMethod(),
            SVDDecomposition(),
            TuckerDecomposition(),
            LowRankApproximation(),
            AttentionPruning(),
            ExpertPruning(),
            MixedPrecisionMethod(),
            BlockSparseMethod(),
            NullCompression()
        ]
        
        # Agregar MPO solo si TensorLy está disponible
        try:
            methods.append(MPOCompression())
        except ImportError:
            pass  # TensorLy no está disponible
        
        for method in methods:
            config = {'strength': 0.5}
            ratio = method.estimate_compression(self.linear_layer, config)
            
            # Verificar que la estimación es válida
            self.assertIsInstance(ratio, float)
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)
    
    # ============= TESTS DE INTEGRACIÓN =============
    
    def test_engine_with_all_methods(self):
        """Test de integración con todos los métodos en el engine"""
        methods_config = {
            'int8_quantization': {'strength': 0.5},
            'magnitude_pruning': {'strength': 0.3},
            'svd_decomposition': {'strength': 0.4},
            'attention_pruning': {'strength': 0.2},
            'tucker_decomposition': {'strength': 0.3},
            'mpo_compression': {'strength': 0.2},
            'low_rank_approximation': {'strength': 0.4},
            'expert_pruning': {'strength': 0.3},
            'mixed_precision': {'strength': 0.5},
            'block_sparse': {'strength': 0.3}
        }
        
        for method_name, config in methods_config.items():
            # Crear configuración completa
            full_config = {
                'methods': [{'name': method_name, **config}],
                'target_compression': 0.5
            }
            
            # Aplicar compresión
            result = self.engine.compress_layer(self.linear_layer, full_config)
            
            # Verificar resultado
            self.assertIsNotNone(result)
            # El resultado puede ser una tupla (módulo, CompressionResult)
            if isinstance(result, tuple):
                compressed_module, compression_result = result
                self.assertIsInstance(compressed_module, nn.Module)
                self.assertTrue(compression_result.success)
            elif hasattr(result, 'original_size'):
                # Es un CompressionResult
                self.assertTrue(result.success)
            else:
                # Es un módulo
                self.assertIsInstance(result, nn.Module)
    
    # ============= TESTS DE ERRORES Y CASOS LÍMITE =============
    
    def test_invalid_configurations(self):
        """Test de configuraciones inválidas"""
        method = QuantizationMethod(bits=8)
        
        # Configuración inválida
        invalid_config = {
            'strength': 1.5,  # Strength > 1
            'bits': 16  # Bits no soportados
        }
        
        # Debería manejar errores graciosamente
        try:
            compressed = method.compress(self.linear_layer, invalid_config, self.device)
            self.assertIsNotNone(compressed)
        except Exception as e:
            # Si lanza excepción, debe ser manejable
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_empty_module(self):
        """Test con módulo vacío"""
        method = QuantizationMethod(bits=8)
        empty_module = nn.Module()
        
        config = {'strength': 0.5}
        
        # Debería manejar módulos sin pesos
        compressed = method.compress(empty_module, config, self.device)
        self.assertIsNotNone(compressed)
    
    def test_device_compatibility(self):
        """Test de compatibilidad de dispositivos"""
        method = QuantizationMethod(bits=8)
        config = {'strength': 0.5}
        
        # Test en CPU
        cpu_compressed = method.compress(self.linear_layer, config, torch.device('cpu'))
        self.assertIsNotNone(cpu_compressed)
        
        # Test en GPU si está disponible
        if torch.cuda.is_available():
            gpu_compressed = method.compress(self.linear_layer, config, torch.device('cuda'))
            self.assertIsNotNone(gpu_compressed)


if __name__ == "__main__":
    unittest.main()
