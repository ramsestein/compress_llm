"""
Detección automática de arquitecturas de modelos
"""
from typing import Dict, Optional
from transformers import AutoConfig

class ArchitectureDetector:
    """Detecta y caracteriza arquitecturas de modelos"""
    
    # Configuraciones de arquitecturas conocidas
    ARCHITECTURES = {
        'llama': {
            'attention_patterns': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'ffn_patterns': ['gate_proj', 'up_proj', 'down_proj'],
            'norm_type': 'RMSNorm',
            'attention_type': 'standard'
        },
        'mistral': {
            'attention_patterns': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'ffn_patterns': ['gate_proj', 'up_proj', 'down_proj'],
            'norm_type': 'RMSNorm',
            'attention_type': 'gqa',
            'special_features': ['sliding_window']
        },
        'mixtral': {
            'attention_patterns': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'ffn_patterns': ['w1', 'w2', 'w3'],
            'norm_type': 'RMSNorm',
            'attention_type': 'gqa',
            'moe_type': 'sparse',
            'num_experts': 8,
            'experts_per_token': 2
        },
        'deepseek': {
            'attention_patterns': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'ffn_patterns': ['gate_proj', 'up_proj', 'down_proj'],
            'norm_type': 'RMSNorm',
            'attention_type': 'standard',
            'moe_type': 'sparse_with_shared',
            'num_experts': 64,
            'num_shared_experts': 2
        },
        'qwen': {
            'attention_patterns': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'ffn_patterns': ['gate_proj', 'up_proj', 'down_proj'],
            'norm_type': 'RMSNorm',
            'attention_type': 'standard',
            'special_features': ['bias_attention']
        }
    }
    
    @classmethod
    def detect(cls, model_config: AutoConfig) -> tuple[str, Dict]:
        """
        Detecta la arquitectura del modelo
        
        Returns:
            tuple: (nombre_arquitectura, configuración)
        """
        model_type = getattr(model_config, 'model_type', '').lower()
        architectures = getattr(model_config, 'architectures', [])
        
        # Detección por model_type
        for arch_name in cls.ARCHITECTURES.keys():
            if arch_name in model_type:
                return arch_name, cls.ARCHITECTURES[arch_name]
        
        # Detección por architectures list
        for arch in architectures:
            arch_lower = arch.lower()
            for arch_name in cls.ARCHITECTURES.keys():
                if arch_name in arch_lower:
                    return arch_name, cls.ARCHITECTURES[arch_name]
        
        # Detección de características especiales
        if hasattr(model_config, 'num_experts'):
            return 'mixtral', cls.ARCHITECTURES['mixtral']
        
        # Default
        return 'llama', cls.ARCHITECTURES['llama']
    
    @classmethod
    def get_model_stats(cls, model_config: AutoConfig, architecture: str) -> Dict:
        """Extrae estadísticas del modelo basadas en la configuración"""
        stats = {
            'architecture': architecture,
            'total_layers': getattr(model_config, 'num_hidden_layers', 0),
            'hidden_size': getattr(model_config, 'hidden_size', 0),
            'num_attention_heads': getattr(model_config, 'num_attention_heads', 0),
            'vocab_size': getattr(model_config, 'vocab_size', 0),
        }
        
        # MoE específico
        if architecture in ['mixtral', 'deepseek']:
            stats['is_moe'] = True
            stats['num_experts'] = getattr(model_config, 'num_local_experts', 
                                          getattr(model_config, 'num_experts', 8))
            stats['experts_per_token'] = getattr(model_config, 'num_experts_per_tok', 2)
        
        # GQA específico
        if architecture in ['mistral', 'mixtral']:
            stats['num_kv_heads'] = getattr(model_config, 'num_key_value_heads', 
                                           stats['num_attention_heads'])
            stats['sliding_window'] = getattr(model_config, 'sliding_window', None)
        
        return stats