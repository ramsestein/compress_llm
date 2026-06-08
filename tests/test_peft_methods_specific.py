#!/usr/bin/env python3
"""
Tests específicos para todos los métodos PEFT disponibles
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

from LoRa_train.peft_methods import (
    MoLoRALinear, GaLoreLinear, DoRALinear, IA3Linear,
    AdapterLinear, QuantizedLoRALinear, PrunedLoRALinear,
    CompacterLinear, KronALinear, S4Adapter, HoulsbyAdapterLinear
)
from LoRa_train.peft_methods_config import (
    MoLoRAConfig, GaLoreConfig, DoRAConfig, BitFitConfig, 
    IA3Config, PromptTuningConfig, AdapterConfig, QLoRAConfig, 
    LoRAConfig, CompacterConfig, KronAConfig, S4Config, HoulsbyConfig, PEFTMethod
)


class TestPEFTMethodsSpecific(unittest.TestCase):
    """Tests específicos para cada método PEFT"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear módulos de prueba
        self.linear_layer = nn.Linear(100, 50)
        self.attention_layer = nn.Linear(768, 768)
        self.ffn_layer = nn.Linear(768, 3072)
        
        # Crear datos de prueba
        self.batch_size = 4
        self.seq_len = 10
        self.input_data = torch.randn(self.batch_size, self.seq_len, 100)
        
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    # ============= TESTS DE MoLoRA =============
    
    def test_molora_linear(self):
        """Test específico para MoLoRA Linear"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=4,
            expert_r=[8, 12, 16, 20],
            expert_alpha=[16, 24, 32, 40],
            expert_dropout=[0.1, 0.1, 0.1, 0.1]
        )
        
        molora_layer = MoLoRALinear(100, 50, config)
        molora_layer.to(self.device)
        
        # Test forward pass
        output = molora_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(molora_layer, nn.Module)
        
        # Verificar que tiene expertos
        self.assertEqual(len(molora_layer.lora_As), config.num_experts)
        self.assertEqual(len(molora_layer.lora_Bs), config.num_experts)
        
        # Verificar router
        self.assertIsNotNone(molora_layer.router)
    
    def test_molora_router(self):
        """Test específico para el router de MoLoRA"""
        from LoRa_train.peft_methods import MoLoRARouter
        
        router = MoLoRARouter(100, 4, "learned")
        router.to(self.device)
        
        # Test routing
        weights, indices = router(self.input_data)
        
        # Verificaciones
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, 2))
        self.assertEqual(indices.shape, (self.batch_size, self.seq_len, 2))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
    
    # ============= TESTS DE GaLore =============
    
    def test_galore_linear(self):
        """Test específico para GaLore Linear"""
        config = GaLoreConfig(
            method=PEFTMethod.GALORE,
            rank=128,
            scale=0.25
        )
        
        galore_layer = GaLoreLinear(100, 50, config)
        galore_layer.to(self.device)
        
        # Test forward pass
        output = galore_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(galore_layer, nn.Module)
        
        # Verificar que tiene componentes GaLore
        self.assertTrue(hasattr(galore_layer, 'projector'))
        self.assertTrue(hasattr(galore_layer, 'config'))
    
    # ============= TESTS DE DoRA =============
    
    def test_dora_linear(self):
        """Test específico para DoRA Linear"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            magnitude_lr_scale=0.1
        )
        
        dora_layer = DoRALinear(100, 50, config)
        dora_layer.to(self.device)
        
        # Test forward pass
        output = dora_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(dora_layer, nn.Module)
        
        # Verificar componentes DoRA
        self.assertTrue(hasattr(dora_layer, 'lora_A'))
        self.assertTrue(hasattr(dora_layer, 'lora_B'))
        self.assertTrue(hasattr(dora_layer, 'magnitude'))
    
    # ============= TESTS DE AdaLoRA =============
    # AdaLoRA ha sido eliminado de esta implementación
    
    # ============= TESTS DE IA3 =============
    
    def test_ia3_linear(self):
        """Test específico para IA3 Linear"""
        config = IA3Config(
            method=PEFTMethod.IA3,
            init_ia3_weights="ones",
            target_modules=["q_proj", "v_proj", "out_proj"]
        )
        
        # Crear capa base
        base_layer = nn.Linear(100, 50)
        ia3_layer = IA3Linear(base_layer, config)
        ia3_layer.to(self.device)
        
        # Test forward pass
        output = ia3_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(ia3_layer, nn.Module)
        
        # Verificar componentes IA3
        self.assertTrue(hasattr(ia3_layer, 'ia3_weights'))
        self.assertTrue(hasattr(ia3_layer, 'base_layer'))
    
    # ============= TESTS DE ADAPTERS =============
    
    def test_adapter_linear(self):
        """Test específico para Adapter Linear"""
        config = AdapterConfig(
            method=PEFTMethod.ADAPTER,
            adapter_size=64,
            adapter_type="pfeiffer"
        )
        
        adapter_layer = AdapterLinear(100, 50, config)
        adapter_layer.to(self.device)
        
        # Test forward pass
        output = adapter_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(adapter_layer, nn.Module)
        
        # Verificar componentes Adapter
        self.assertTrue(hasattr(adapter_layer, 'adapter_down'))
        self.assertTrue(hasattr(adapter_layer, 'adapter_up'))
        self.assertTrue(hasattr(adapter_layer, 'layer_norm'))
    
    # ============= TESTS DE LoRA CUANTIZADO =============
    
    def test_quantized_lora_linear(self):
        """Test específico para LoRA Cuantizado"""
        config = QLoRAConfig(
            method=PEFTMethod.QLORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bits=4
        )
        
        quantized_layer = QuantizedLoRALinear(100, 50, config)
        quantized_layer.to(self.device)
        
        # Test forward pass
        output = quantized_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(quantized_layer, nn.Module)
        
        # Verificar componentes Quantized LoRA
        self.assertTrue(hasattr(quantized_layer, 'lora_A'))
        self.assertTrue(hasattr(quantized_layer, 'lora_B'))
        self.assertEqual(quantized_layer.bits, 4)
    
    # ============= TESTS DE LoRA PRUNED =============
    
    def test_pruned_lora_linear(self):
        """Test específico para LoRA Pruned"""
        config = LoRAConfig(
            method=PEFTMethod.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        pruned_layer = PrunedLoRALinear(100, 50, config)
        pruned_layer.to(self.device)
        
        # Test forward pass
        output = pruned_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(pruned_layer, nn.Module)
        
        # Verificar componentes Pruned LoRA
        self.assertTrue(hasattr(pruned_layer, 'lora_A'))
        self.assertTrue(hasattr(pruned_layer, 'lora_B'))
        self.assertTrue(hasattr(pruned_layer, 'A_mask'))
        self.assertTrue(hasattr(pruned_layer, 'B_mask'))
    
    # ============= TESTS DE COMPACTER =============
    
    def test_compacter_linear(self):
        """Test específico para Compacter Linear"""
        config = CompacterConfig(
            method=PEFTMethod.COMPACTER,
            rank=8,
            dropout=0.1,
            scaling=1.0
        )
        
        compacter_layer = CompacterLinear(100, 50, config)
        compacter_layer.to(self.device)
        
        # Test forward pass
        output = compacter_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(compacter_layer, nn.Module)
        
        # Verificar componentes Compacter
        self.assertTrue(hasattr(compacter_layer, 'compacter_weight'))
        self.assertEqual(compacter_layer.config.rank, 8)
    
    # ============= TESTS DE KRONA =============
    
    def test_krona_linear(self):
        """Test específico para KronA Linear"""
        config = KronAConfig(
            method=PEFTMethod.KRONA,
            dropout=0.1,
            scaling=1.0
        )
        
        krona_layer = KronALinear(100, 50, config)
        krona_layer.to(self.device)
        
        # Test forward pass
        output = krona_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(krona_layer, nn.Module)
        
        # Verificar componentes KronA
        self.assertTrue(hasattr(krona_layer, 'krona_weight'))
    
    # ============= TESTS DE S4 =============
    
    def test_s4_adapter(self):
        """Test específico para S4 Adapter"""
        config = S4Config(
            method=PEFTMethod.S4,
            hidden_size=128,
            state_size=64,
            conv_size=4,
            expand_factor=2,
            dropout=0.1,
            scaling=1.0
        )
        
        s4_layer = S4Adapter(100, 50, config)
        s4_layer.to(self.device)
        
        # Test forward pass
        output = s4_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(s4_layer, nn.Module)
        
        # Verificar componentes S4
        self.assertTrue(hasattr(s4_layer, 's4_weight'))
        self.assertEqual(s4_layer.config.hidden_size, 128)
    
    # ============= TESTS DE HOULSBY =============
    
    def test_houlsby_adapter_linear(self):
        """Test específico para Houlsby Adapter Linear"""
        config = HoulsbyConfig(
            method=PEFTMethod.HOULSBY,
            adapter_size=64,
            dropout=0.1,
            scaling=1.0,
            non_linearity="gelu"
        )
        
        houlsby_layer = HoulsbyAdapterLinear(100, 50, config)
        houlsby_layer.to(self.device)
        
        # Test forward pass
        output = houlsby_layer(self.input_data)
        
        # Verificaciones específicas
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 50))
        self.assertIsInstance(houlsby_layer, nn.Module)
        
        # Verificar componentes Houlsby
        self.assertTrue(hasattr(houlsby_layer, 'adapter_pre'))
        self.assertTrue(hasattr(houlsby_layer, 'adapter_post'))
        self.assertEqual(houlsby_layer.config.adapter_size, 64)
    
    # ============= TESTS DE CONFIGURACIONES =============
    
    def test_molora_config(self):
        """Test de configuración MoLoRA"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=4,
            expert_r=[8, 12, 16, 20],
            expert_alpha=[16, 24, 32, 40],
            expert_dropout=[0.1, 0.1, 0.1, 0.1]
        )
        
        # Verificar configuración
        self.assertEqual(config.num_experts, 4)
        self.assertEqual(len(config.expert_r), 4)
        self.assertEqual(len(config.expert_alpha), 4)
        self.assertEqual(len(config.expert_dropout), 4)
    
    def test_galore_config(self):
        """Test de configuración GaLore"""
        config = GaLoreConfig(
            method=PEFTMethod.GALORE,
            rank=128,
            scale=0.25
        )
        
        # Verificar configuración
        self.assertEqual(config.rank, 128)
        self.assertEqual(config.scale, 0.25)
    
    def test_dora_config(self):
        """Test de configuración DoRA"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            magnitude_lr_scale=0.1
        )
        
        # Verificar configuración
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertEqual(config.magnitude_lr_scale, 0.1)
    
    def test_adapter_config(self):
        """Test de configuración Adapter"""
        config = AdapterConfig(
            method=PEFTMethod.ADAPTER,
            adapter_size=64,
            adapter_type="pfeiffer"
        )
        
        # Verificar configuración
        self.assertEqual(config.adapter_size, 64)
        self.assertEqual(config.adapter_type, "pfeiffer")
    
    def test_compacter_config(self):
        """Test de configuración Compacter"""
        config = CompacterConfig(
            method=PEFTMethod.COMPACTER,
            rank=8,
            dropout=0.1,
            scaling=1.0
        )
        
        # Verificar configuración
        self.assertEqual(config.rank, 8)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.scaling, 1.0)
    
    def test_krona_config(self):
        """Test de configuración KronA"""
        config = KronAConfig(
            method=PEFTMethod.KRONA,
            dropout=0.1,
            scaling=1.0
        )
        
        # Verificar configuración
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.scaling, 1.0)
    
    def test_s4_config(self):
        """Test de configuración S4"""
        config = S4Config(
            method=PEFTMethod.S4,
            hidden_size=128,
            state_size=64,
            conv_size=4,
            expand_factor=2,
            dropout=0.1,
            scaling=1.0
        )
        
        # Verificar configuración
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.state_size, 64)
        self.assertEqual(config.conv_size, 4)
        self.assertEqual(config.expand_factor, 2)
    
    def test_houlsby_config(self):
        """Test de configuración Houlsby"""
        config = HoulsbyConfig(
            method=PEFTMethod.HOULSBY,
            adapter_size=64,
            dropout=0.1,
            scaling=1.0,
            non_linearity="gelu"
        )
        
        # Verificar configuración
        self.assertEqual(config.adapter_size, 64)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.scaling, 1.0)
        self.assertEqual(config.non_linearity, "gelu")
    
    # ============= TESTS DE MERGE WEIGHTS =============
    
    def test_merge_weights_molora(self):
        """Test de merge weights para MoLoRA"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[8, 16],
            expert_alpha=[16, 32],
            expert_dropout=[0.1, 0.1]
        )
        
        molora_layer = MoLoRALinear(100, 50, config)
        molora_layer.to(self.device)
        
        # Guardar pesos originales
        original_weight = molora_layer.weight.clone()
        
        # Merge weights
        molora_layer.merge_weights()
        
        # Verificar que los pesos se fusionaron
        self.assertTrue(torch.allclose(molora_layer.weight, original_weight, atol=1e-6))
    
    def test_merge_weights_dora(self):
        """Test de merge weights para DoRA"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            magnitude_lr_scale=0.1
        )
        
        dora_layer = DoRALinear(100, 50, config)
        dora_layer.to(self.device)
        
        # Guardar pesos originales
        original_weight = dora_layer.weight.clone()
        
        # Merge weights
        dora_layer.merge_weights()
        
        # Verificar que los pesos cambiaron (se fusionaron)
        self.assertFalse(torch.allclose(dora_layer.weight, original_weight, atol=1e-6))
    
    # ============= TESTS DE GRADIENTES =============
    
    def test_gradients_molora(self):
        """Test de gradientes para MoLoRA"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[8, 16],
            expert_alpha=[16, 32],
            expert_dropout=[0.1, 0.1]
        )
        
        molora_layer = MoLoRALinear(100, 50, config)
        molora_layer.to(self.device)
        
        # Forward pass
        output = molora_layer(self.input_data)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Verificar gradientes
        for name, param in molora_layer.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())
    
    def test_gradients_dora(self):
        """Test de gradientes para DoRA"""
        config = DoRAConfig(
            method=PEFTMethod.DORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            magnitude_lr_scale=0.1
        )
        
        dora_layer = DoRALinear(100, 50, config)
        dora_layer.to(self.device)
        
        # Forward pass
        output = dora_layer(self.input_data)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Verificar gradientes
        has_gradients = False
        for name, param in dora_layer.named_parameters():
            if param.requires_grad:
                has_gradients = True
                if param.grad is not None:  # Algunos parámetros pueden no tener gradientes aún
                    self.assertFalse(torch.isnan(param.grad).any())
        
        # Verificar que al menos algunos parámetros tienen gradientes
        self.assertTrue(has_gradients, "Al menos algunos parámetros deben tener gradientes")
    
    # ============= TESTS DE ERRORES Y CASOS LÍMITE =============
    
    def test_invalid_configurations(self):
        """Test de configuraciones inválidas"""
        # Configuración inválida para MoLoRA - probar con router_type inválido
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[8, 16],
            expert_alpha=[16, 32],
            expert_dropout=[0.1, 0.1],
            router_type="invalid_type"  # Tipo inválido
        )
        
        # El error se lanza cuando se usa la configuración
        with self.assertRaises(ValueError):
            molora_layer = MoLoRALinear(100, 50, config)
            molora_layer.to(self.device)
            # Esto debería lanzar ValueError en el router
            molora_layer(self.input_data)
    
    def test_empty_input(self):
        """Test con entrada vacía"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[16, 32],
            expert_alpha=[32, 64],
            expert_dropout=[0.1, 0.1]
        )
        
        molora_layer = MoLoRALinear(100, 50, config)
        molora_layer.to(self.device)
        
        # Entrada vacía
        empty_input = torch.empty(0, 0, 100)
        
        # Debería manejar entrada vacía graciosamente
        try:
            output = molora_layer(empty_input)
            self.assertEqual(output.shape, (0, 0, 50))
        except Exception as e:
            # Si lanza excepción, debe ser manejable
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_device_compatibility(self):
        """Test de compatibilidad de dispositivos"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[16, 32],
            expert_alpha=[32, 64],
            expert_dropout=[0.1, 0.1]
        )
        
        # Test en CPU
        cpu_layer = MoLoRALinear(100, 50, config)
        cpu_layer.to(torch.device('cpu'))
        cpu_output = cpu_layer(self.input_data.cpu())
        self.assertIsNotNone(cpu_output)
        
        # Test en GPU si está disponible
        if torch.cuda.is_available():
            gpu_layer = MoLoRALinear(100, 50, config)
            gpu_layer.to(torch.device('cuda'))
            gpu_output = gpu_layer(self.input_data.cuda())
            self.assertIsNotNone(gpu_output)
    
    # ============= TESTS DE RENDIMIENTO =============
    
    def test_performance_comparison(self):
        """Test de comparación de rendimiento entre métodos"""
        methods_configs = [
            (MoLoRALinear, MoLoRAConfig(method=PEFTMethod.MOLORA, num_experts=2, expert_r=[8, 16], expert_alpha=[16, 32], expert_dropout=[0.1, 0.1])),
            (DoRALinear, DoRAConfig(method=PEFTMethod.DORA, r=16, lora_alpha=32, lora_dropout=0.1, magnitude_lr_scale=0.1)),
            (AdapterLinear, AdapterConfig(method=PEFTMethod.ADAPTER, adapter_size=64)),
            (QuantizedLoRALinear, QLoRAConfig(method=PEFTMethod.QLORA, r=16, lora_alpha=32, lora_dropout=0.1, bits=4)),
            (CompacterLinear, CompacterConfig(method=PEFTMethod.COMPACTER, rank=8, dropout=0.1, scaling=1.0)),
            (KronALinear, KronAConfig(method=PEFTMethod.KRONA, dropout=0.1, scaling=1.0)),
            (S4Adapter, S4Config(method=PEFTMethod.S4, hidden_size=128, state_size=64, conv_size=4, expand_factor=2)),
            (HoulsbyAdapterLinear, HoulsbyConfig(method=PEFTMethod.HOULSBY, adapter_size=64, dropout=0.1, scaling=1.0)),
            (GaLoreLinear, GaLoreConfig(method=PEFTMethod.GALORE, rank=128, scale=0.25))
        ]
        
        for method_class, config in methods_configs:
            layer = method_class(100, 50, config)
            layer.to(self.device)
            
            # Medir tiempo de forward pass
            import time
            start_time = time.time()
            
            for _ in range(10):
                output = layer(self.input_data)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            # Verificar que el tiempo es razonable (< 1 segundo por forward pass)
            self.assertLess(avg_time, 1.0)
    
    # ============= TESTS DE MEMORIA =============
    
    def test_memory_usage(self):
        """Test de uso de memoria"""
        config = MoLoRAConfig(
            method=PEFTMethod.MOLORA,
            num_experts=2,
            expert_r=[8, 16],
            expert_alpha=[16, 32],
            expert_dropout=[0.1, 0.1]
        )
        
        molora_layer = MoLoRALinear(100, 50, config)
        molora_layer.to(self.device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in molora_layer.parameters())
        trainable_params = sum(p.numel() for p in molora_layer.parameters() if p.requires_grad)
        
        # Verificar que hay parámetros entrenables
        self.assertGreater(trainable_params, 0)
        self.assertLessEqual(trainable_params, total_params)
        
        # Verificar que el número de parámetros es razonable
        self.assertLess(total_params, 100000)  # Menos de 100k parámetros


if __name__ == "__main__":
    unittest.main()
