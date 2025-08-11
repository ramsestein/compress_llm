"""
Estrategias de compresión para diferentes casos de uso
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CompressionMethod:
    """Método de compresión con configuración"""
    name: str
    compression_ratio: float
    parameters: Dict

class CompressionStrategies:
    """Gestiona estrategias de compresión por caso de uso"""
    
    def __init__(self, layer_profiles: Dict, model_stats: Dict):
        self.layer_profiles = layer_profiles
        self.model_stats = model_stats
    
    def get_strategy(self, use_case: str) -> Dict:
        """Obtiene estrategia para un caso de uso específico"""
        strategies = {
            'rag': self._rag_strategy,
            'ner': self._ner_strategy,
            'chatbot': self._chatbot_strategy,
            'agent': self._agent_strategy
        }
        
        if use_case not in strategies:
            raise ValueError(f"Caso de uso no soportado: {use_case}")
        
        return strategies[use_case]()
    
    def _rag_strategy(self) -> Dict:
        """Estrategia para Retrieval-Augmented Generation"""
        return self._create_strategy(
            name='RAG',
            description='Optimizado para comprensión de contexto y síntesis',
            compression_map={
                'embedding': 0.1,
                'attention': {'early': 0.25, 'late': 0.45},
                'ffn': 0.65,
                'output': 0.35
            },
            expected_performance=0.95
        )
    
    def _ner_strategy(self) -> Dict:
        """Estrategia para Named Entity Recognition"""
        return self._create_strategy(
            name='NER',
            description='Máxima precisión en clasificación de tokens',
            compression_map={
                'embedding': 0.0,  # No tocar
                'attention': {'early': 0.15, 'late': 0.75},
                'ffn': {'early': 0.35, 'late': 0.75},
                'output': 0.40
            },
            expected_performance=0.97
        )
    
    def _chatbot_strategy(self) -> Dict:
        """Estrategia para Chatbot conversacional"""
        return self._create_strategy(
            name='Chatbot',
            description='Balance entre comprensión y generación',
            compression_map={
                'embedding': 0.15,
                'attention': 0.35,  # Uniforme
                'ffn': {'early': 0.35, 'late': 0.55},
                'output': 0.30
            },
            expected_performance=0.93
        )
    
    def _agent_strategy(self) -> Dict:
        """Estrategia para Agente con herramientas"""
        return self._create_strategy(
            name='Agent',
            description='Preservar reasoning y parsing preciso',
            compression_map={
                'embedding': 0.05,
                'attention': {'early': 0.25, 'mid': 0.15, 'late': 0.45},
                'ffn': {'early': 0.25, 'late': 0.45},
                'output': 0.30
            },
            expected_performance=0.94
        )
    
    def _create_strategy(self, name: str, description: str, 
                        compression_map: Dict, expected_performance: float) -> Dict:
        """Crea una estrategia con los parámetros dados"""
        strategy = {
            'name': name,
            'description': description,
            'layer_strategies': {},
            'expected_performance': expected_performance
        }
        
        total_size = 0
        compressed_size = 0
        
        for layer_name, profile in self.layer_profiles.items():
            compression_ratio = self._get_compression_ratio(
                profile, compression_map
            )
            
            methods = self._recommend_methods(profile, compression_ratio)
            
            strategy['layer_strategies'][layer_name] = {
                'compression_ratio': compression_ratio,
                'methods': methods,
                'priority': self._get_priority(profile, compression_ratio)
            }
            
            total_size += profile.size_mb
            compressed_size += profile.size_mb * (1 - compression_ratio)
        
        strategy['expected_compression'] = 1 - (compressed_size / total_size) if total_size > 0 else 0
        strategy['final_size_mb'] = compressed_size
        
        return strategy
    
    def _get_compression_ratio(self, profile, compression_map: Dict) -> float:
        """Determina ratio de compresión basado en el mapa"""
        layer_type = profile.type
        
        if layer_type in compression_map:
            ratio_config = compression_map[layer_type]
            
            # Si es un diccionario, usar posición relativa
            if isinstance(ratio_config, dict):
                if profile.relative_position < 0.3:
                    return ratio_config.get('early', 0.3)
                elif profile.relative_position < 0.7:
                    return ratio_config.get('mid', ratio_config.get('early', 0.4))
                else:
                    return ratio_config.get('late', 0.5)
            else:
                return ratio_config
        
        # Default
        return 0.3
    
    def _recommend_methods(self, profile, compression_ratio: float) -> List[Tuple[str, float]]:
        """Recomienda métodos específicos para alcanzar el ratio objetivo"""
        methods = []
        
        # Distribución típica de métodos según ratio total
        if compression_ratio < 0.2:
            # Compresión ligera
            methods.append(('int8_quantization', compression_ratio))
        elif compression_ratio < 0.5:
            # Compresión moderada
            if profile.rank_ratio > 0.7:
                methods.append(('mpo', compression_ratio * 0.7))
                methods.append(('int8_quantization', compression_ratio * 0.3))
            else:
                methods.append(('pruning', compression_ratio * 0.6))
                methods.append(('int8_quantization', compression_ratio * 0.4))
        else:
            # Compresión agresiva
            if profile.type == 'attention':
                methods.append(('tucker', compression_ratio * 0.5))
                methods.append(('pruning', compression_ratio * 0.3))
                methods.append(('int4_quantization', compression_ratio * 0.2))
            else:
                methods.append(('pruning', compression_ratio * 0.6))
                methods.append(('int4_quantization', compression_ratio * 0.4))
        
        return methods
    
    def _get_priority(self, profile, compression_ratio: float) -> str:
        """Determina prioridad de la capa"""
        # Alta importancia + alta compresión = crítico
        if profile.relative_position < 0.3 and compression_ratio < 0.3:
            return 'critical'
        elif compression_ratio > 0.6:
            return 'low'
        else:
            return 'medium'