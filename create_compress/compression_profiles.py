"""
Perfiles de compresión predefinidos y optimizados para diferentes casos de uso
"""
from typing import Dict, List, Any, Optional

# Perfiles optimizados predefinidos
COMPRESSION_PROFILES = {
    'conservative': {
        'name': 'Conservative',
        'description': 'Compresión mínima para máxima calidad - Recomendado para producción',
        'goal': 'max_quality',
        'target_compression': 0.3,
        'profile': 'conservative',  # Añadir campo profile
        'layer_configs': {
            'attention': {
                'methods': [
                    {'name': 'head_pruning', 'strength': 0.2},
                    {'name': 'int8_quantization', 'strength': 0.4}
                ],
                'total_compression_ratio': 0.25
            },
            'ffn': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.3},
                    {'name': 'int8_quantization', 'strength': 0.5}
                ],
                'total_compression_ratio': 0.35
            },
            'embedding': {
                'methods': [
                    {'name': 'int8_quantization', 'strength': 0.3}
                ],
                'total_compression_ratio': 0.15
            },
            'output': {
                'methods': [
                    {'name': 'none', 'strength': 0.0}
                ],
                'total_compression_ratio': 0.0
            },
            'normalization': {
                'methods': [
                    {'name': 'none', 'strength': 0.0}
                ],
                'total_compression_ratio': 0.0
            }
        },
        'preserve_patterns': ['lm_head', 'embed_tokens', 'word_embeddings'],
        'final_layers_special': True,
        'risk_level': 'low'
    },
    
    'balanced': {
        'name': 'Balanced',
        'description': 'Balance óptimo entre tamaño y calidad - Uso general',
        'goal': 'balanced',
        'target_compression': 0.5,
        'profile': 'balanced',  # Añadir campo profile
        'layer_configs': {
            'attention': {
                'methods': [
                    {'name': 'attention_pruning', 'strength': 0.4},
                    {'name': 'low_rank_approximation', 'strength': 0.3},
                    {'name': 'int8_quantization', 'strength': 0.6}
                ],
                'total_compression_ratio': 0.45
            },
            'ffn': {
                'methods': [
                    {'name': 'structured_pruning', 'strength': 0.5},
                    {'name': 'int8_quantization', 'strength': 0.7}
                ],
                'total_compression_ratio': 0.55
            },
            'embedding': {
                'methods': [
                    {'name': 'int8_quantization', 'strength': 0.5}
                ],
                'total_compression_ratio': 0.25
            },
            'output': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.2}
                ],
                'total_compression_ratio': 0.15
            },
            'linear': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.5},
                    {'name': 'int8_quantization', 'strength': 0.6}
                ],
                'total_compression_ratio': 0.5
            },
            'other': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.4}
                ],
                'total_compression_ratio': 0.3
            }
        },
        'position_modifiers': {
            'early_layers': {
                'range': [0.0, 0.3],
                'compression_multiplier': 0.7,
                'description': 'Reducir compresión en capas iniciales'
            },
            'middle_layers': {
                'range': [0.3, 0.7],
                'compression_multiplier': 1.0,
                'description': 'Compresión normal'
            },
            'late_layers': {
                'range': [0.7, 1.0],
                'compression_multiplier': 1.2,
                'description': 'Mayor compresión en capas finales'
            }
        },
        'final_layers_special': True,
        'risk_level': 'medium'
    },
    
    'aggressive': {
        'name': 'Aggressive',
        'description': 'Máxima compresión - Para experimentos o recursos muy limitados',
        'goal': 'max_compression',
        'target_compression': 0.7,
        'profile': 'aggressive',  # Añadir campo profile
        'layer_configs': {
            'attention': {
                'methods': [
                    {'name': 'attention_pruning', 'strength': 0.7},
                    {'name': 'tensor_decomposition', 'strength': 0.6},
                    {'name': 'int4_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.7
            },
            'ffn': {
                'methods': [
                    {'name': 'structured_pruning', 'strength': 0.75},
                    {'name': 'int4_quantization', 'strength': 0.85}
                ],
                'total_compression_ratio': 0.75
            },
            'embedding': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.3},
                    {'name': 'int8_quantization', 'strength': 0.7}
                ],
                'total_compression_ratio': 0.4
            },
            'output': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.4},
                    {'name': 'int8_quantization', 'strength': 0.6}
                ],
                'total_compression_ratio': 0.35
            },
            'linear': {
                'methods': [
                    {'name': 'structured_pruning', 'strength': 0.7},
                    {'name': 'int4_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.7
            },
            'other': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.6},
                    {'name': 'int8_quantization', 'strength': 0.7}
                ],
                'total_compression_ratio': 0.6
            }
        },
        'position_modifiers': {
            'early_layers': {
                'range': [0.0, 0.2],
                'compression_multiplier': 0.8,
                'description': 'Proteger capas muy iniciales'
            },
            'middle_layers': {
                'range': [0.2, 0.8],
                'compression_multiplier': 1.1,
                'description': 'Compresión aumentada'
            },
            'late_layers': {
                'range': [0.8, 1.0],
                'compression_multiplier': 1.3,
                'description': 'Máxima compresión en capas finales'
            }
        },
        'final_layers_special': False,
        'risk_level': 'high'
    },
    
    'mobile': {
        'name': 'Mobile Deployment',
        'description': 'Optimizado para dispositivos móviles - Balance entre tamaño y latencia',
        'goal': 'mobile_deployment',
        'target_compression': 0.6,
        'profile': 'mobile',  # Añadir campo profile
        'layer_configs': {
            'attention': {
                'methods': [
                    {'name': 'head_pruning', 'strength': 0.5},
                    {'name': 'int8_quantization', 'strength': 0.9}  # INT8 es eficiente en móviles
                ],
                'total_compression_ratio': 0.55
            },
            'ffn': {
                'methods': [
                    {'name': 'structured_pruning', 'strength': 0.6},  # Estructurado para SIMD
                    {'name': 'int8_quantization', 'strength': 0.9}
                ],
                'total_compression_ratio': 0.65
            },
            'embedding': {
                'methods': [
                    {'name': 'int8_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.4
            },
            'output': {
                'methods': [
                    {'name': 'int8_quantization', 'strength': 0.7}
                ],
                'total_compression_ratio': 0.35
            },
            'other': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.5}
                ],
                'total_compression_ratio': 0.4
            }
        },
        'preserve_patterns': ['lm_head'],
        'final_layers_special': True,
        'risk_level': 'medium'
    },
    
    'edge': {
        'name': 'Edge Deployment',
        'description': 'Para dispositivos edge con recursos muy limitados',
        'goal': 'edge_deployment',
        'target_compression': 0.75,
        'profile': 'edge',  # Añadir campo profile
        'layer_configs': {
            'attention': {
                'methods': [
                    {'name': 'attention_pruning', 'strength': 0.8},
                    {'name': 'int4_quantization', 'strength': 0.9}
                ],
                'total_compression_ratio': 0.75
            },
            'ffn': {
                'methods': [
                    {'name': 'structured_pruning', 'strength': 0.8},
                    {'name': 'int4_quantization', 'strength': 0.95}
                ],
                'total_compression_ratio': 0.8
            },
            'embedding': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.5},
                    {'name': 'int4_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.6
            },
            'output': {
                'methods': [
                    {'name': 'structured_pruning', 'strength': 0.5},
                    {'name': 'int8_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.5
            },
            'other': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.7}
                ],
                'total_compression_ratio': 0.6
            }
        },
        'final_layers_special': False,
        'risk_level': 'high'
    },
    
    'research': {
        'name': 'Research',
        'description': 'Configuración experimental para investigación',
        'goal': 'research',
        'target_compression': 0.8,
        'profile': 'research',  # Añadir campo profile
        'layer_configs': {
            'attention': {
                'methods': [
                    {'name': 'tensor_decomposition', 'strength': 0.8},
                    {'name': 'attention_pruning', 'strength': 0.7},
                    {'name': 'int4_quantization', 'strength': 0.9}
                ],
                'total_compression_ratio': 0.8
            },
            'ffn': {
                'methods': [
                    {'name': 'low_rank_approximation', 'strength': 0.7},
                    {'name': 'structured_pruning', 'strength': 0.8},
                    {'name': 'int4_quantization', 'strength': 0.95}
                ],
                'total_compression_ratio': 0.85
            },
            'embedding': {
                'methods': [
                    {'name': 'tensor_decomposition', 'strength': 0.6},
                    {'name': 'int4_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.65
            },
            'output': {
                'methods': [
                    {'name': 'low_rank_approximation', 'strength': 0.6},
                    {'name': 'int4_quantization', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.6
            },
            'other': {
                'methods': [
                    {'name': 'magnitude_pruning', 'strength': 0.8}
                ],
                'total_compression_ratio': 0.7
            }
        },
        'position_modifiers': {
            'early_layers': {
                'range': [0.0, 0.1],
                'compression_multiplier': 0.5,
                'description': 'Mínima compresión en primeras capas'
            },
            'middle_layers': {
                'range': [0.1, 0.9],
                'compression_multiplier': 1.2,
                'description': 'Compresión experimental aumentada'
            },
            'late_layers': {
                'range': [0.9, 1.0],
                'compression_multiplier': 0.8,
                'description': 'Proteger última capa'
            }
        },
        'final_layers_special': True,
        'risk_level': 'high'
    }
}

# Funciones auxiliares
def get_profile(name: str) -> Optional[Dict[str, Any]]:
    """Obtiene un perfil por nombre"""
    return COMPRESSION_PROFILES.get(name)

def list_profiles() -> List[str]:
    """Lista todos los perfiles disponibles"""
    return list(COMPRESSION_PROFILES.keys())

def get_profile_info(name: str) -> Dict[str, Any]:
    """Obtiene información resumida de un perfil"""
    profile = get_profile(name)
    if not profile:
        return None
    
    return {
        'name': profile.get('name', name),
        'description': profile.get('description', ''),
        'target_compression': profile.get('target_compression', 0),
        'risk_level': profile.get('risk_level', 'unknown'),
        'num_layer_types': len(profile.get('layer_configs', {})),
        'has_position_modifiers': profile.get('position_modifiers') is not None,
        'preserves_critical_layers': len(profile.get('preserve_patterns', [])) > 0
    }

def create_custom_profile(name: str, 
                         base_profile: str = 'balanced',
                         modifications: Dict[str, Any] = None) -> Dict[str, Any]:
    """Crea un perfil personalizado basado en uno existente"""
    base = get_profile(base_profile)
    if not base:
        raise ValueError(f"Perfil base no encontrado: {base_profile}")
    
    # Copiar configuración base
    custom = {
        'name': name,
        'description': f"Perfil personalizado basado en {base_profile}",
        'goal': base.get('goal', 'custom'),
        'target_compression': base.get('target_compression', 0.5),
        'profile': 'custom',
        'layer_configs': base.get('layer_configs', {}).copy(),
        'position_modifiers': base.get('position_modifiers', {}).copy() if base.get('position_modifiers') else None,
        'preserve_patterns': base.get('preserve_patterns', []).copy(),
        'final_layers_special': base.get('final_layers_special', False),
        'risk_level': base.get('risk_level', 'medium')
    }
    
    # Aplicar modificaciones
    if modifications:
        custom.update(modifications)
    
    return custom

def estimate_model_size(original_size_mb: float, profile_name: str) -> Dict[str, float]:
    """Estima el tamaño final del modelo con un perfil"""
    profile = get_profile(profile_name)
    if not profile:
        return None
    
    target_compression = profile.get('target_compression', 0)
    compressed_size = original_size_mb * (1 - target_compression)
    
    return {
        'original_size_mb': original_size_mb,
        'compressed_size_mb': compressed_size,
        'compression_ratio': target_compression,
        'size_reduction_mb': original_size_mb - compressed_size,
        'size_reduction_percent': target_compression * 100
    }

# Recomendaciones basadas en el tamaño del modelo
def recommend_profile(model_size_mb: float, 
                     use_case: str = 'general',
                     hardware: str = 'gpu') -> str:
    """Recomienda un perfil basado en características del modelo"""
    
    # Por caso de uso
    if use_case == 'production':
        return 'conservative'
    elif use_case == 'mobile':
        return 'mobile'
    elif use_case == 'edge':
        return 'edge'
    elif use_case == 'research':
        return 'research'
    
    # Por tamaño y hardware
    if hardware == 'mobile':
        return 'mobile' if model_size_mb > 1000 else 'balanced'
    elif hardware == 'edge':
        return 'edge'
    elif hardware == 'cpu':
        if model_size_mb < 500:
            return 'conservative'
        elif model_size_mb < 2000:
            return 'balanced'
        else:
            return 'aggressive'
    else:  # GPU
        if model_size_mb < 1000:
            return 'conservative'
        elif model_size_mb < 5000:
            return 'balanced'
        else:
            return 'aggressive'

# Validación de configuración
def validate_profile(profile: Dict[str, Any]) -> List[str]:
    """Valida que un perfil esté correctamente configurado"""
    issues = []
    
    # Validar target_compression
    target = profile.get('target_compression', -1)
    if not 0 < target < 1:
        issues.append("target_compression debe estar entre 0 y 1")
    
    # Validar layer_configs
    layer_configs = profile.get('layer_configs', {})
    if not layer_configs:
        issues.append("layer_configs no puede estar vacío")
    
    # Validar métodos
    for layer_type, config in layer_configs.items():
        if 'methods' not in config:
            issues.append(f"Falta 'methods' en configuración de {layer_type}")
        elif not config['methods']:
            issues.append(f"Lista de métodos vacía para {layer_type}")
        
        if 'total_compression_ratio' not in config:
            issues.append(f"Falta 'total_compression_ratio' en {layer_type}")
    
    return issues