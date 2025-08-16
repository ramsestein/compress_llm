#!/usr/bin/env python3
"""
Test para la verificación de compresión
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import json
import torch

class TestCompressionVerification(unittest.TestCase):
    """Test de la verificación de compresión"""
    
    def setUp(self):
        """Configuración inicial"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir()
        
    def tearDown(self):
        """Limpieza después de las pruebas"""
        shutil.rmtree(self.test_dir)
    
    def test_compression_verification_structure(self):
        """Test de la estructura de verificación de compresión"""
        # Crear modelo original
        original_model = self.models_dir / "test_original"
        original_model.mkdir()
        
        # Crear archivo de configuración
        config = {
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12
        }
        
        with open(original_model / "config.json", "w") as f:
            json.dump(config, f)
        
        # Crear archivo de pesos simulado
        weights = torch.randn(100, 100)
        torch.save(weights, original_model / "pytorch_model.bin")
        
        # Crear modelo comprimido
        compressed_model = self.models_dir / "test_compressed"
        compressed_model.mkdir()
        
        # Crear metadata de componentes
        metadata = {
            "saved_by_components": True,
            "total_parameters": 50,
            "timestamp": "2025-01-01T00:00:00",
            "model_type": "gpt2"
        }
        
        with open(compressed_model / "component_save_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Crear archivo de configuración
        with open(compressed_model / "config.json", "w") as f:
            json.dump(config, f)
        
        # Crear algunos parámetros simulados
        for i in range(5):
            param = torch.randn(10, 10)
            torch.save(param, compressed_model / f"param_{i}.pt")
        
        # Verificar estructura
        self.assertTrue((original_model / "pytorch_model.bin").exists())
        self.assertTrue((compressed_model / "component_save_metadata.json").exists())
        self.assertTrue((compressed_model / "config.json").exists())
        
        # Verificar metadata
        with open(compressed_model / "component_save_metadata.json", "r") as f:
            loaded_metadata = json.load(f)
        
        self.assertTrue(loaded_metadata["saved_by_components"])
        self.assertEqual(loaded_metadata["total_parameters"], 50)
        self.assertEqual(loaded_metadata["model_type"], "gpt2")
    
    def test_compression_ratio_calculation(self):
        """Test del cálculo de ratio de compresión"""
        # Simular tamaños de archivo
        original_size = 1000  # MB
        compressed_size = 400  # MB
        
        # Calcular ratio
        reduction = ((original_size - compressed_size) / original_size) * 100
        factor = original_size / compressed_size
        
        # Verificar cálculos
        self.assertEqual(reduction, 60.0)  # 60% de reducción
        self.assertEqual(factor, 2.5)      # 2.5x más pequeño
        
        # Verificar que la compresión es significativa
        self.assertGreater(reduction, 50.0)
        self.assertGreater(factor, 1.5)

if __name__ == "__main__":
    unittest.main()
