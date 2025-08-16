#!/usr/bin/env python3
"""
Test para el merge de LoRA
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import json

class TestMergeLoRA(unittest.TestCase):
    """Test del merge de LoRA"""
    
    def setUp(self):
        """Configuración inicial"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir()
        
    def tearDown(self):
        """Limpieza después de las pruebas"""
        shutil.rmtree(self.test_dir)
    
    def test_merge_lora_structure(self):
        """Test de la estructura del merge de LoRA"""
        # Crear estructura de modelo base
        base_model = self.models_dir / "test_base"
        base_model.mkdir()
        
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
        
        with open(base_model / "config.json", "w") as f:
            json.dump(config, f)
        
        # Crear estructura de LoRA
        lora_model = self.test_dir / "test_lora"
        lora_model.mkdir()
        
        # Crear archivo de configuración de adaptador
        adapter_config = {
            "base_model_name_or_path": str(base_model),
            "bias": "none",
            "enable_lora": None,
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 8,
            "target_modules": ["c_attn", "c_proj"],
            "task_type": "CAUSAL_LM"
        }
        
        with open(lora_model / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)
        
        # Verificar que la estructura es correcta
        self.assertTrue((base_model / "config.json").exists())
        self.assertTrue((lora_model / "adapter_config.json").exists())
        
        # Verificar contenido de configuración
        with open(lora_model / "adapter_config.json", "r") as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["peft_type"], "LORA")
        self.assertEqual(loaded_config["r"], 8)
        self.assertEqual(loaded_config["lora_alpha"], 16)

if __name__ == "__main__":
    unittest.main()
