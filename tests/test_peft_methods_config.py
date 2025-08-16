#!/usr/bin/env python3
"""
Tests para PEFT methods configuration
"""
import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import LoRa_train modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from LoRa_train.peft_methods_config import (
    PEFTMethod, BasePEFTConfig, LoRAConfig, MoLoRAConfig, GaLoreConfig,
    DoRAConfig, AdaLoRAConfig, BitFitConfig, IA3Config, PromptTuningConfig,
    AdapterConfig, QLoRAConfig, PEFTPresets, get_config_by_name,
    get_available_methods, get_available_presets, estimate_memory_usage
)


class TestPEFTMethodsConfig(unittest.TestCase):
    """Tests para PEFT methods configuration"""
    
    def test_peft_method_enum(self):
        """Test para enum PEFTMethod"""
        # Verificar que todos los métodos están definidos
        expected_methods = [
            'lora', 'molora', 'galore', 'dora', 'adalora',
            'bitfit', 'ia3', 'prompt_tuning', 'adapter', 'qlora'
        ]
        
        for method_name in expected_methods:
            self.assertTrue(hasattr(PEFTMethod, method_name.upper()))
            method = getattr(PEFTMethod, method_name.upper())
            self.assertEqual(method.value, method_name)
    
    def test_base_peft_config(self):
        """Test para BasePEFTConfig"""
        config = BasePEFTConfig(
            method=PEFTMethod.LORA,
            learning_rate=1e-4,
            num_train_epochs=5,
            per_device_train_batch_size=8
        )
        
        # Verificar valores por defecto
        self.assertEqual(config.method, PEFTMethod.LORA)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.per_device_train_batch_size, 8)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.warmup_steps, 100)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.logging_steps, 10)
        self.assertEqual(config.save_steps, 500)
        self.assertEqual(config.eval_steps, 500)
        self.assertEqual(config.save_total_limit, 3)
        self.assertEqual(config.load_best_model_at_end, False)
        self.assertEqual(config.metric_for_best_model, "eval_loss")
        self.assertEqual(config.greater_is_better, False)
        self.assertEqual(config.seed, 42)
    
    def test_lora_config(self):
        """Test para LoRAConfig"""
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="all",
            target_modules=["attn.c_attn", "attn.c_proj"],
            learning_rate=2e-4
        )
        
        # Verificar valores específicos de LoRA
        self.assertEqual(config.method, PEFTMethod.LORA)
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertEqual(config.bias, "all")
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
        self.assertEqual(config.learning_rate, 2e-4)
        
        # Verificar post_init
        self.assertEqual(config.method, PEFTMethod.LORA)
    
    def test_molora_config(self):
        """Test para MoLoRAConfig"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=4,
            expert_r=[8, 8, 16, 16],
            expert_alpha=[16, 16, 32, 32],
            expert_dropout=[0.1, 0.1, 0.2, 0.2],
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        
        # Verificar valores específicos de MoLoRA
        self.assertEqual(config.method, PEFTMethod.MOLORA)
        self.assertEqual(config.num_experts, 4)
        self.assertEqual(config.expert_r, [8, 8, 16, 16])
        self.assertEqual(config.expert_alpha, [16, 16, 32, 32])
        self.assertEqual(config.expert_dropout, [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
        
        # Verificar post_init con valores por defecto
        default_config = MoLoRAConfig(method=PEFTMethod.MOLORA, num_experts=2)
        self.assertEqual(default_config.expert_r, [8, 8])
        self.assertEqual(default_config.expert_alpha, [16, 16])
        self.assertEqual(default_config.expert_dropout, [0.1, 0.1])
    
    def test_galore_config(self):
        """Test para GaLoreConfig"""
        config = GaLoreConfig(
            method=PEFTMethod.GALORE,
            rank=64,
            scale=0.25,
            target_modules=["attn.c_attn", "attn.c_proj"],
            learning_rate=5e-5
        )
        
        # Verificar valores específicos de GaLore
        self.assertEqual(config.method, PEFTMethod.GALORE)
        self.assertEqual(config.rank, 64)
        self.assertEqual(config.scale, 0.25)
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
        self.assertEqual(config.learning_rate, 5e-5)
    
    def test_dora_config(self):
        """Test para DoRAConfig"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            magnitude_lr_scale=0.1,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        
        # Verificar valores específicos de DoRA
        self.assertEqual(config.method, PEFTMethod.DORA)
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertEqual(config.magnitude_lr_scale, 0.1)
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
    
    def test_adalora_config(self):
        """Test para AdaLoRAConfig"""
        config = AdaLoRAConfig(
            method=PEFTMethod.ADALORA,
            init_r=64,
            target_r=16,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            beta1=0.85,
            beta2=0.85,
            target_modules=["attn.c_attn", "attn.c_proj"]
        )
        
        # Verificar valores específicos de AdaLoRA
        self.assertEqual(config.method, PEFTMethod.ADALORA)
        self.assertEqual(config.init_r, 64)
        self.assertEqual(config.target_r, 16)
        self.assertEqual(config.tinit, 200)
        self.assertEqual(config.tfinal, 1000)
        self.assertEqual(config.deltaT, 10)
        self.assertEqual(config.beta1, 0.85)
        self.assertEqual(config.beta2, 0.85)
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
    
    def test_bitfit_config(self):
        """Test para BitFitConfig"""
        config = BitFitConfig(
            method=PEFTMethod.BITFIT,
            train_embeddings=True,
            train_layer_norms=False,
            target_modules=["embeddings", "layer_norm"],
            learning_rate=1e-5
        )
        
        # Verificar valores específicos de BitFit
        self.assertEqual(config.method, PEFTMethod.BITFIT)
        self.assertEqual(config.train_embeddings, True)
        self.assertEqual(config.train_layer_norms, False)
        self.assertEqual(config.target_modules, ["embeddings", "layer_norm"])
        self.assertEqual(config.learning_rate, 1e-5)
    
    def test_ia3_config(self):
        """Test para IA3Config"""
        config = IA3Config(
            method=PEFTMethod.IA3,
            init_ia3_weights="ones",
            target_modules=["attn.c_attn", "attn.c_proj"],
            learning_rate=1e-3
        )
        
        # Verificar valores específicos de IA3
        self.assertEqual(config.method, PEFTMethod.IA3)
        self.assertEqual(config.init_ia3_weights, "ones")
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
        self.assertEqual(config.learning_rate, 1e-3)
    
    def test_prompt_tuning_config(self):
        """Test para PromptTuningConfig"""
        config = PromptTuningConfig(
            method=PEFTMethod.PROMPT_TUNING,
            num_virtual_tokens=20,
            prompt_tuning_init="random",
            target_modules=["embeddings"],
            learning_rate=1e-3
        )
        
        # Verificar valores específicos de Prompt Tuning
        self.assertEqual(config.method, PEFTMethod.PROMPT_TUNING)
        self.assertEqual(config.num_virtual_tokens, 20)
        self.assertEqual(config.prompt_tuning_init, "random")
        self.assertEqual(config.target_modules, ["embeddings"])
        self.assertEqual(config.learning_rate, 1e-3)
    
    def test_adapter_config(self):
        """Test para AdapterConfig"""
        config = AdapterConfig(
            method=PEFTMethod.ADAPTER,
            adapter_size=64,
            adapter_type="pfeiffer",
            target_modules=["attn.c_attn", "attn.c_proj"],
            learning_rate=1e-4
        )
        
        # Verificar valores específicos de Adapter
        self.assertEqual(config.method, PEFTMethod.ADAPTER)
        self.assertEqual(config.adapter_size, 64)
        self.assertEqual(config.adapter_type, "pfeiffer")
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
        self.assertEqual(config.learning_rate, 1e-4)
    
    def test_qlora_config(self):
        """Test para QLoRAConfig"""
        config = QLoRAConfig(
            method=PEFTMethod.QLORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bits=4,
            target_modules=["attn.c_attn", "attn.c_proj"],
            learning_rate=1e-4
        )
        
        # Verificar valores específicos de QLoRA
        self.assertEqual(config.method, PEFTMethod.QLORA)
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.target_modules, ["attn.c_attn", "attn.c_proj"])
        self.assertEqual(config.learning_rate, 1e-4)
    
    def test_peft_presets(self):
        """Test para PEFTPresets"""
        # Verificar que todos los presets están definidos
        expected_presets = ["efficient", "balanced", "quality", "memory_efficient"]
        
        for preset_name in expected_presets:
            self.assertIn(preset_name, PEFTPresets)
            preset = PEFTPresets[preset_name]
            self.assertIsInstance(preset, dict)
            self.assertGreater(len(preset), 0)
        
        # Verificar preset "efficient"
        efficient_preset = PEFTPresets["efficient"]
        self.assertIn("lora", efficient_preset)
        self.assertIn("bitfit", efficient_preset)
        self.assertIn("ia3", efficient_preset)
        
        # Verificar que las configuraciones tienen el método correcto
        self.assertEqual(efficient_preset["lora"].method, PEFTMethod.LORA)
        self.assertEqual(efficient_preset["bitfit"].method, PEFTMethod.BITFIT)
        self.assertEqual(efficient_preset["ia3"].method, PEFTMethod.IA3)
    
    def test_get_config_by_name(self):
        """Test para get_config_by_name"""
        # Obtener configuración existente
        lora_config = get_config_by_name("efficient", "lora")
        self.assertIsInstance(lora_config, LoRAConfig)
        self.assertEqual(lora_config.method, PEFTMethod.LORA)
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)
        
        # Verificar error para preset inexistente
        with self.assertRaises(ValueError):
            get_config_by_name("nonexistent", "lora")
        
        # Verificar error para método inexistente
        with self.assertRaises(ValueError):
            get_config_by_name("efficient", "nonexistent")
    
    def test_get_available_methods(self):
        """Test para get_available_methods"""
        # Obtener métodos disponibles para preset "efficient"
        methods = get_available_methods("efficient")
        expected_methods = ["lora", "bitfit", "ia3"]
        
        for method in expected_methods:
            self.assertIn(method, methods)
        
        # Verificar preset inexistente
        methods = get_available_methods("nonexistent")
        self.assertEqual(methods, [])
    
    def test_get_available_presets(self):
        """Test para get_available_presets"""
        presets = get_available_presets()
        expected_presets = ["efficient", "balanced", "quality", "memory_efficient"]
        
        for preset in expected_presets:
            self.assertIn(preset, presets)
    
    def test_estimate_memory_usage(self):
        """Test para estimate_memory_usage"""
        # Test para LoRA
        lora_config = LoRAConfig(method=PEFTMethod.LORA, r=8, lora_alpha=16)
        memory_estimate = estimate_memory_usage(lora_config, 1.0)  # 1B parameters
        
        self.assertIn("model_memory_gb", memory_estimate)
        self.assertIn("peft_memory_mb", memory_estimate)
        self.assertIn("total_memory_gb", memory_estimate)
        
        # Verificar que la memoria PEFT es menor que la del modelo
        self.assertLess(memory_estimate["peft_memory_mb"], memory_estimate["model_memory_gb"] * 1024)
        
        # Test para BitFit
        bitfit_config = BitFitConfig(method=PEFTMethod.BITFIT)
        bitfit_memory = estimate_memory_usage(bitfit_config, 1.0)
        self.assertIn("peft_memory_mb", bitfit_memory)
        
        # Test para Prompt Tuning
        prompt_config = PromptTuningConfig(method=PEFTMethod.PROMPT_TUNING, num_virtual_tokens=20)
        prompt_memory = estimate_memory_usage(prompt_config, 1.0)
        self.assertIn("peft_memory_mb", prompt_memory)
        
        # Test para QLoRA
        qlora_config = QLoRAConfig(method=PEFTMethod.QLORA, r=8, lora_alpha=16, bits=4)
        qlora_memory = estimate_memory_usage(qlora_config, 1.0)
        self.assertIn("peft_memory_mb", qlora_memory)
    
    def test_config_serialization(self):
        """Test para serialización de configuraciones"""
        # Crear configuración
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=16,
            lora_alpha=32,
            target_modules=["attn.c_attn"]
        )
        
        # Convertir a diccionario
        config_dict = {
            'method': config.method.value,
            'r': config.r,
            'lora_alpha': config.lora_alpha,
            'target_modules': config.target_modules
        }
        
        # Verificar que se puede recrear
        new_config = LoRAConfig(
            method=PEFTMethod(config_dict['method']),
            r=config_dict['r'],
            lora_alpha=config_dict['lora_alpha'],
            target_modules=config_dict['target_modules']
        )
        
        self.assertEqual(new_config.r, config.r)
        self.assertEqual(new_config.lora_alpha, config.lora_alpha)
        self.assertEqual(new_config.target_modules, config.target_modules)


if __name__ == "__main__":
    unittest.main()
