"""
Perfilado y análisis de capas individuales
"""
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re

@dataclass
class LayerProfile:
    """Información completa de una capa"""
    name: str
    type: str
    size_mb: float
    parameters: int
    position: int
    relative_position: float
    
    # Métricas
    gradient_importance: float = 0.0
    activation_variance: float = 0.0
    weight_magnitude: float = 0.0
    sparsity: float = 0.0
    rank_ratio: float = 0.0
    
    # Estructura
    input_dim: int = 0
    output_dim: int = 0
    compression_potential: float = 0.0
    
    # MoE
    is_moe: bool = False
    num_experts: Optional[int] = None
    expert_utilization: Optional[List[float]] = None
    
    # GQA
    is_gqa: bool = False
    num_kv_heads: Optional[int] = None
    
    # Recomendaciones
    recommended_methods: List[Tuple[str, float]] = None

    def __post_init__(self):
        """Inicializa valores por defecto si no se proporcionaron"""
        if self.recommended_methods is None:
            self.recommended_methods = []

class LayerProfiler:
    """Analiza y perfila capas de redes neuronales"""
    
    def __init__(self, architecture_config: Dict):
        self.arch_config = architecture_config
    
    def profile_layer(self, name: str, module: torch.nn.Module, 
                     model_stats: Dict) -> LayerProfile:
        """Perfila una capa individual"""
        # Información básica
        params = sum(p.numel() for p in module.parameters())
        size_mb = params * 2 / 1024 / 1024  # FP16
        
        # Clasificar tipo
        layer_type = self._classify_layer(name, module)
        
        # Posición
        position = self._extract_position(name)
        total_layers = model_stats.get('total_layers', 1)
        relative_position = position / max(total_layers, 1)
        
        # Análisis de pesos
        weight_stats = self._analyze_weights(module)
        
        # Dimensiones
        input_dim, output_dim = self._get_dimensions(module)
        
        # Crear perfil base
        profile = LayerProfile(
            name=name,
            type=layer_type,
            size_mb=size_mb,
            parameters=params,
            position=position,
            relative_position=relative_position,
            weight_magnitude=weight_stats['magnitude'],
            sparsity=weight_stats['sparsity'],
            rank_ratio=weight_stats['rank_ratio'],
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # Agregar info específica de arquitectura
        self._add_architecture_specific_info(profile, name, model_stats)
        
        # Calcular potencial de compresión
        profile.compression_potential = self.calculate_compression_potential(profile)
        
        # Recomendar métodos
        profile.recommended_methods = self._recommend_compression_methods(profile)
        
        return profile
    
    def _classify_layer(self, name: str, module: torch.nn.Module) -> str:
        """Clasifica el tipo de capa"""
        name_lower = name.lower()
        
        # MoE
        if 'expert' in name_lower:
            return 'moe_expert' if 'router' not in name_lower else 'moe_router'
        
        # Embeddings
        if isinstance(module, torch.nn.Embedding) or 'embed' in name_lower:
            return 'embedding'
        
        # Atención
        attention_patterns = self.arch_config.get('attention_patterns', [])
        if any(p in name_lower for p in attention_patterns):
            return 'attention'
        
        # FFN
        ffn_patterns = self.arch_config.get('ffn_patterns', [])
        if any(p in name_lower for p in ffn_patterns):
            return 'ffn'
        
        # Output
        if any(x in name_lower for x in ['lm_head', 'output', 'classifier']):
            return 'output'
        
        # Normalización: ampliar patrones y detección por tipo de módulo
        #
        # Durante el análisis del modelo original se clasificaban como
        # "normalization" únicamente aquellas capas que contenían la cadena
        # "norm" en su nombre. Modelos como GPT‑2 usan la nomenclatura
        # "ln_1", "ln_2", etc., que quedaban sin detectar y por lo tanto no
        # se incluían en la configuración de compresión. Al aplicar la
        # compresión, otro clasificador sí identificaba estas capas y se
        # generaban advertencias del tipo "No hay configuración para tipo
        # 'normalization'".  Esto hacía que las capas de normalización se
        # preservaran de forma implícita y se mostraran advertencias ruidosas.

        # Para unificar la detección entre el análisis y la compresión, se
        # amplían los patrones reconocidos e incluso se verifica el tipo del
        # módulo. De esta forma cualquier capa de normalización será
        # correctamente etiquetada y aparecerá en la configuración generada.
        if isinstance(
            module,
            (
                torch.nn.LayerNorm,
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.GroupNorm,
            ),
        ):
            return "normalization"

        norm_patterns = ["norm", "ln", "layernorm", "batchnorm", "groupnorm"]
        if any(p in name_lower for p in norm_patterns):
            return "normalization"
        
        return 'other'
    
    def _extract_position(self, name: str) -> int:
        """Extrae número de capa del nombre"""
        patterns = [
            r'layers\.(\d+)', 
            r'layer\.(\d+)', 
            r'blocks\.(\d+)', 
            r'h\.(\d+)',
            r'transformer\.h\.(\d+)',
            r'encoder\.layer\.(\d+)',
            r'decoder\.layer\.(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        
        return 0
    
    def _analyze_weights(self, module: torch.nn.Module) -> Dict:
        """Análisis de pesos mejorado"""
        stats = {
            'magnitude': 0.0,
            'sparsity': 0.0,
            'rank_ratio': 0.0,
        }
        
        if hasattr(module, 'weight'):
            weight = module.weight.data
            
            # Magnitud
            stats['magnitude'] = weight.abs().mean().item()
            
            # Sparsidad adaptativa
            if weight.numel() > 0:
                threshold = 0.01 * weight.abs().max().item()
                stats['sparsity'] = (weight.abs() < threshold).float().mean().item()
            
            # Análisis SVD mejorado
            if weight.dim() == 2 and min(weight.shape) > 1:
                try:
                    # Para matrices muy grandes, usar aproximación
                    if weight.numel() > 1e7:  # 10M parámetros
                        # Usar randomized SVD
                        k = min(100, min(weight.shape) // 2)
                        U, S, V = torch.svd_lowrank(weight, q=k)
                    else:
                        U, S, V = torch.svd(weight, compute_uv=False)
                    
                    # Calcular ratio de energía
                    if len(S) > 0:
                        total_energy = S.sum().item()
                        k = min(20, len(S) // 4)
                        top_k_energy = S[:k].sum().item()
                        stats['rank_ratio'] = top_k_energy / total_energy if total_energy > 0 else 0
                except Exception as e:
                    # En caso de error SVD, usar valor por defecto
                    stats['rank_ratio'] = 0.5
        
        return stats
    
    def _get_dimensions(self, module: torch.nn.Module) -> Tuple[int, int]:
        """Obtiene dimensiones de entrada/salida"""
        # Linear layers
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            return module.in_features, module.out_features
        
        # Convolutional layers
        elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            return module.in_channels, module.out_channels
        
        # Embedding layers
        elif isinstance(module, torch.nn.Embedding):
            return module.num_embeddings, module.embedding_dim
        
        # LayerNorm, BatchNorm, etc.
        elif hasattr(module, 'normalized_shape'):
            if isinstance(module.normalized_shape, tuple):
                dim = module.normalized_shape[0] if module.normalized_shape else 0
            else:
                dim = module.normalized_shape
            return dim, dim
        
        # Generic weight-based dimension inference
        elif hasattr(module, 'weight'):
            shape = module.weight.shape
            if len(shape) >= 2:
                return shape[1], shape[0]
            elif len(shape) == 1:
                return shape[0], shape[0]
        
        # Default
        return 0, 0
    
    def _add_architecture_specific_info(self, profile: LayerProfile, 
                                       name: str, model_stats: Dict):
        """Agrega información específica de la arquitectura"""
        # MoE
        if model_stats.get('is_moe') and profile.type in ['moe_expert', 'ffn']:
            profile.is_moe = True
            profile.num_experts = model_stats.get('num_experts', 8)
            # Inicializar utilización uniforme por defecto
            if profile.num_experts:
                profile.expert_utilization = [1.0 / profile.num_experts] * profile.num_experts
        
        # GQA
        if model_stats.get('num_kv_heads') and profile.type == 'attention':
            if 'k_proj' in name or 'v_proj' in name:
                profile.is_gqa = True
                profile.num_kv_heads = model_stats['num_kv_heads']
    
    def calculate_compression_potential(self, profile: LayerProfile) -> float:
        """Calcula el potencial de compresión de una capa"""
        potential = 0.0
        
        # Factores que aumentan compresibilidad
        potential += profile.sparsity * 0.3
        potential += profile.rank_ratio * 0.3
        
        # Posición (capas finales más comprimibles)
        if profile.type in ['attention', 'ffn']:
            potential += profile.relative_position * 0.2
        
        # Tamaño (capas grandes son mejores candidatas)
        size_factor = min(profile.size_mb / 100, 1.0)
        potential += size_factor * 0.1
        
        # Estabilidad (baja varianza = más comprimible)
        if profile.activation_variance > 0:
            stability = 1.0 / (1.0 + profile.activation_variance)
            potential += stability * 0.1
        
        # Ajustes por tipo de capa
        if profile.type == 'embedding':
            potential *= 0.5  # Los embeddings son críticos
        elif profile.type == 'output':
            potential *= 0.7  # La capa de salida es importante
        elif profile.type == 'normalization':
            potential *= 0.3  # Normalización es muy importante
        
        return min(potential, 1.0)
    
    def _recommend_compression_methods(self, profile: LayerProfile) -> List[Tuple[str, float]]:
        """Recomienda métodos de compresión basados en el perfil de la capa"""
        recommendations = []
        
        # MPO/Tucker para capas con alta correlación
        if profile.rank_ratio > 0.7:
            if profile.type == 'attention':
                recommendations.append(('mpo', 0.9))
            else:
                recommendations.append(('tucker', 0.85))
        
        # Pruning para capas con alta sparsidad
        if profile.sparsity > 0.3:
            recommendations.append(('pruning', 0.85))
        
        # Quantization basada en posición y tipo
        if profile.type not in ['embedding', 'normalization']:
            if profile.relative_position < 0.3:  # Capas tempranas
                recommendations.append(('int8_quantization', 0.7))
            else:  # Capas tardías
                recommendations.append(('int4_quantization', 0.8))
        
        # Para capas MoE
        if profile.is_moe:
            if profile.type == 'moe_router':
                recommendations.append(('int8_quantization', 0.5))
            else:
                recommendations.append(('expert_pruning', 0.9))
        
        # Ordenar por confianza
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:3]  # Retornar top 3