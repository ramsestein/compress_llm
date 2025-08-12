""""
Analizador detallado de modelos para estrategias de compresión
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import warnings

from .compression_strategies import CompressionStrategies

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class LayerProfile:
    """Perfil optimizado de una capa"""
    name: str
    type: str
    position: str  # 'early', 'middle', 'late'
    relative_position: float
    parameters: int
    size_mb: float
    input_dim: int = 0
    output_dim: int = 0
    
    # Métricas de importancia (calculadas bajo demanda)
    gradient_importance: float = 0.0
    activation_variance: float = 0.0
    weight_magnitude: float = 0.0
    sparsity: float = 0.0
    rank_ratio: float = 1.0
    compression_potential: float = 0.0
    
    # Flags especiales
    is_critical: bool = False
    is_moe: bool = False
    num_experts: int = 0
    
    # Cache de cálculos
    _metrics_computed: bool = field(default=False, init=False)

class OptimizedLayerProfiler:
    """Analizador optimizado de capas individuales"""
    
    def __init__(self):
        self._cache = {}
        self._type_cache = {}
    
    @lru_cache(maxsize=1000)
    def _get_layer_type(self, name: str, module_class: str) -> str:
        """Determina tipo de capa con cache"""
        name_lower = name.lower()
        
        # Clasificación rápida por nombre
        if 'attn' in name_lower or 'attention' in name_lower:
            return 'attention'
        elif 'mlp' in name_lower or 'ffn' in name_lower or 'feed_forward' in name_lower:
            return 'ffn'
        elif 'embed' in name_lower:
            return 'embedding'
        elif 'norm' in name_lower or 'ln' in name_lower:
            return 'normalization'
        elif 'lm_head' in name_lower or 'classifier' in name_lower:
            return 'output'
        elif 'expert' in name_lower:
            return 'moe_expert'
        elif 'router' in name_lower or 'gate' in name_lower:
            return 'moe_router'
        
        # Por tipo de módulo
        if 'Linear' in module_class:
            return 'linear'
        elif 'LayerNorm' in module_class or 'BatchNorm' in module_class:
            return 'normalization'
        elif 'Embedding' in module_class:
            return 'embedding'
        
        return 'other'
    
    def profile_layer(self, name: str, module: nn.Module, 
                     model_stats: Dict[str, Any]) -> LayerProfile:
        """Perfila una capa con optimizaciones"""
        # Check cache
        cache_key = f"{name}_{id(module)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Información básica
        layer_type = self._get_layer_type(name, module.__class__.__name__)
        
        # Calcular parámetros
        param_count = sum(p.numel() for p in module.parameters())
        size_mb = param_count * 2 / (1024 * 1024)  # FP16
        
        # Posición relativa
        layer_idx = self._extract_layer_index(name)
        num_layers = model_stats.get('num_layers', 100)
        relative_pos = layer_idx / max(num_layers - 1, 1) if layer_idx >= 0 else 0.5
        
        # Determinar posición categórica
        if relative_pos < 0.3:
            position = 'early'
        elif relative_pos < 0.7:
            position = 'middle'
        else:
            position = 'late'
        
        # Dimensiones (si es Linear)
        input_dim, output_dim = 0, 0
        if hasattr(module, 'in_features'):
            input_dim = module.in_features
            output_dim = module.out_features
        elif hasattr(module, 'weight'):
            shape = module.weight.shape
            if len(shape) >= 2:
                output_dim, input_dim = shape[0], shape[1]
        
        # Crear perfil
        profile = LayerProfile(
            name=name,
            type=layer_type,
            position=position,
            relative_position=relative_pos,
            parameters=param_count,
            size_mb=size_mb,
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # Detectar MoE
        if 'expert' in name.lower() or hasattr(module, 'num_experts'):
            profile.is_moe = True
            profile.num_experts = getattr(module, 'num_experts', 8)
        
        # Detectar capas críticas
        if layer_type in ['embedding', 'output', 'moe_router']:
            profile.is_critical = True
        
        # Cache result
        self._cache[cache_key] = profile
        return profile
    
    def _extract_layer_index(self, layer_name: str) -> int:
        """Extrae índice de capa del nombre"""
        import re
        # Buscar patrones como layers.12, layer_12, block.12
        matches = re.findall(r'(?:layers?|blocks?)[._](\d+)', layer_name)
        if matches:
            return int(matches[0])
        return -1
    
    def compute_metrics_batch(self, profiles: List[LayerProfile], 
                             model: nn.Module, 
                             use_sampling: bool = True):
        """Calcula métricas para un lote de perfiles"""
        if not profiles:
            return
        
        # Filtrar solo perfiles que necesitan cálculo
        pending = [p for p in profiles if not p._metrics_computed]
        if not pending:
            return
        
        logger.debug(f"Calculando métricas para {len(pending)} capas")
        
        # Usar ThreadPoolExecutor para paralelizar
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Dividir en lotes
            batch_size = max(1, len(pending) // (mp.cpu_count() * 2))
            batches = [pending[i:i + batch_size] 
                      for i in range(0, len(pending), batch_size)]
            
            # Procesar en paralelo
            futures = []
            for batch in batches:
                future = executor.submit(self._compute_metrics_for_batch, 
                                       batch, model, use_sampling)
                futures.append(future)
            
            # Esperar resultados
            for future in futures:
                future.result()

    def _compute_metrics_for_batch(self, profiles: List[LayerProfile], 
                                   model: nn.Module,
                                   use_sampling: bool):
        """Calcula métricas para un lote de perfiles"""
        for profile in profiles:
            try:
                module = self._get_module_by_name(model, profile.name)
                if module is None:
                    continue
                
                # Solo calcular métricas básicas para eficiencia
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    
                    # Métricas rápidas
                    with torch.no_grad():
                        # Magnitud promedio
                        if use_sampling and weight.numel() > 1e6:
                            # Muestrear para tensores grandes
                            sample_size = 10000
                            flat_weight = weight.flatten()
                            indices = torch.randint(0, flat_weight.size(0), (sample_size,))
                            sample = flat_weight[indices]
                            profile.weight_magnitude = float(sample.abs().mean())
                            profile.sparsity = float((sample == 0).sum()) / sample_size
                        else:
                            profile.weight_magnitude = float(weight.abs().mean())
                            profile.sparsity = float((weight == 0).sum()) / weight.numel()
                        
                        # Rank ratio (aproximado para eficiencia)
                        if weight.dim() == 2 and min(weight.shape) > 10:
                            if use_sampling and weight.numel() > 1e6:
                                # Aproximación rápida
                                profile.rank_ratio = 0.8  # Valor típico
                            else:
                                # SVD rápido en CPU
                                weight_cpu = weight.float().cpu()
                                try:
                                    s = torch.linalg.svdvals(weight_cpu)
                                    threshold = s[0] * 0.01
                                    effective_rank = (s > threshold).sum().item()
                                    profile.rank_ratio = effective_rank / min(weight.shape)
                                except:
                                    profile.rank_ratio = 1.0
                
                # Calcular potencial de compresión
                profile.compression_potential = self._calculate_compression_potential(profile)
                profile._metrics_computed = True
                
            except Exception as e:
                logger.debug(f"Error calculando métricas para {profile.name}: {e}")
                profile._metrics_computed = True
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Obtiene módulo por nombre"""
        try:
            parts = name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            return module
        except:
            return None
    
    def _calculate_compression_potential(self, profile: LayerProfile) -> float:
        """Calcula potencial de compresión basado en métricas"""
        if profile.is_critical:
            return 0.1  # Baja compresión para capas críticas
        
        # Factores que aumentan el potencial de compresión
        factors = []
        
        # Sparsity
        if profile.sparsity > 0.1:
            factors.append(min(profile.sparsity * 2, 0.9))
        
        # Rank ratio
        if profile.rank_ratio < 0.9:
            factors.append(1 - profile.rank_ratio)
        
        # Posición (capas tardías suelen ser más compresibles)
        if profile.position == 'late':
            factors.append(0.3)
        elif profile.position == 'middle':
            factors.append(0.2)
        
        # Tipo de capa
        if profile.type in ['ffn', 'linear']:
            factors.append(0.4)
        elif profile.type == 'attention':
            factors.append(0.3)
        
        # Calcular potencial
        if factors:
            potential = sum(factors) / len(factors)
        else:
            potential = 0.2  # Valor por defecto
        
        return min(potential, 0.95)  # Máximo 95% de compresión

class OptimizedModelAnalyzer:
    """Analizador optimizado de modelos completos"""
    
    def __init__(self, model: nn.Module, model_stats: Dict[str, Any]):
        self.model = model
        self.model_stats = model_stats
        self.profiler = OptimizedLayerProfiler()
        self.layer_profiles: Dict[str, LayerProfile] = {}
        
        # Cache para estrategias
        self._strategy_cache = {}
    
    def analyze(self, calibration_texts: Optional[List[str]] = None,
               quick_mode: bool = False, use_case: str = 'all') -> Dict[str, Any]:
        """Analiza modelo y genera estrategias de compresión"""
        logger.info("🔍 Iniciando análisis optimizado del modelo...")
        
        # 1. Análisis estructural
        self._analyze_structure(quick_mode)
        
        # 2. Calcular métricas (opcional)
        if not quick_mode:
            self._compute_layer_metrics()
        
        # 3. Generar estrategias
        strategies = self._generate_strategies()

        if use_case != 'all':
            compression_strategies = CompressionStrategies(
                self.layer_profiles, 
                self.model_stats
            )
            # Añadir estrategias por caso de uso
            for case in ['rag', 'ner', 'chatbot', 'agent']:
                strategies[case] = compression_strategies.get_strategy(case)
        
        return strategies
    
    def _analyze_structure(self, quick_mode: bool = False):
        """Analiza estructura del modelo"""
        logger.info("📊 Analizando estructura del modelo...")
        
        # Recolectar módulos a analizar
        modules_to_analyze = []
        for name, module in self.model.named_modules():
            if self._should_analyze_layer(name, module):
                modules_to_analyze.append((name, module))
        
        # Modo rápido: analizar solo muestra
        if quick_mode and len(modules_to_analyze) > 50:
            logger.info(f"⚡ Modo rápido: analizando muestra de {len(modules_to_analyze)} capas")
            # Tomar muestra estratificada
            sample_size = 50
            indices = np.linspace(0, len(modules_to_analyze)-1, sample_size, dtype=int)
            modules_to_analyze = [modules_to_analyze[i] for i in indices]
        
        # Analizar módulos
        total_params = 0
        with tqdm(total=len(modules_to_analyze), desc="Analizando capas") as pbar:
            for name, module in modules_to_analyze:
                profile = self.profiler.profile_layer(name, module, self.model_stats)
                self.layer_profiles[name] = profile
                total_params += profile.parameters
                pbar.update(1)
        
        self.model_stats['total_parameters'] = total_params
        self.model_stats['total_size_mb'] = total_params * 2 / (1024 * 1024)
        
        logger.info(f"✅ Analizadas {len(self.layer_profiles)} capas")
        logger.info(f"📦 Tamaño total: {self.model_stats['total_size_mb']:.1f} MB")
    
    def _should_analyze_layer(self, name: str, module: nn.Module) -> bool:
        """Determina si una capa debe ser analizada"""
        # Solo analizar capas con parámetros significativos
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        return param_count > 10000  # Umbral: 10K parámetros
    
    def _compute_layer_metrics(self):
        """Calcula métricas de importancia para las capas"""
        logger.info("📈 Calculando métricas de capas...")
        
        # Convertir a lista para procesamiento por lotes
        profiles_list = list(self.layer_profiles.values())
        
        # Procesar en lotes
        batch_size = 50
        for i in range(0, len(profiles_list), batch_size):
            batch = profiles_list[i:i+batch_size]
            self.profiler.compute_metrics_batch(batch, self.model, use_sampling=True)
    
    def _generate_strategies(self) -> Dict[str, Any]:
        """Genera estrategias de compresión optimizadas"""
        logger.info("🎯 Generando estrategias de compresión...")
        
        strategies = {
            'conservative': self._create_strategy('conservative', 0.3),
            'balanced': self._create_strategy('balanced', 0.5),
            'aggressive': self._create_strategy('aggressive', 0.7),
            'custom': self._create_custom_strategy()
        }
        
        # Agregar recomendación
        strategies['recommended'] = self._get_recommended_strategy()
        
        return strategies
    
    def _create_strategy(self, name: str, target_compression: float) -> Dict[str, Any]:
        """Crea estrategia de compresión con cache"""
        cache_key = f"{name}_{target_compression}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        strategy = {
            'name': name,
            'target_compression': target_compression,
            'layer_configs': defaultdict(dict),
            'estimated_compression': 0,
            'risk_level': 'low' if target_compression < 0.4 else 'medium' if target_compression < 0.6 else 'high'
        }
        
        # Agrupar capas por tipo
        layers_by_type = defaultdict(list)
        for profile in self.layer_profiles.values():
            layers_by_type[profile.type].append(profile)
        
        # Configurar compresión por tipo
        compression_factors = {
            'conservative': {
                'ffn': 0.4, 'attention': 0.3, 'linear': 0.3,
                'embedding': 0.1, 'output': 0.1, 'other': 0.2
            },
            'balanced': {
                'ffn': 0.6, 'attention': 0.5, 'linear': 0.5,
                'embedding': 0.2, 'output': 0.2, 'other': 0.4
            },
            'aggressive': {
                'ffn': 0.8, 'attention': 0.7, 'linear': 0.7,
                'embedding': 0.3, 'output': 0.3, 'other': 0.6
            }
        }
        
        factors = compression_factors.get(name, compression_factors['balanced'])
        
        total_original = 0
        total_compressed = 0
        
        for layer_type, profiles in layers_by_type.items():
            compression_ratio = factors.get(layer_type, 0.5)
            
            # Ajustar por potencial de compresión
            avg_potential = np.mean([p.compression_potential for p in profiles])
            adjusted_ratio = min(compression_ratio * (1 + avg_potential * 0.5), 0.95)
            
            # Seleccionar métodos apropiados
            methods = self._select_compression_methods(layer_type, adjusted_ratio)
            
            strategy['layer_configs'][layer_type] = {
                'compression_ratio': adjusted_ratio,
                'methods': methods,
                'num_layers': len(profiles)
            }
            
            # Calcular tamaños
            type_size = sum(p.size_mb for p in profiles)
            compressed_size = type_size * (1 - adjusted_ratio)
            total_original += type_size
            total_compressed += compressed_size
        
        strategy['estimated_compression'] = 1 - (total_compressed / max(total_original, 1))
        
        # Cache result
        self._strategy_cache[cache_key] = strategy
        return strategy
    
    def _select_compression_methods(self, layer_type: str, 
                                   compression_ratio: float) -> List[Dict[str, Any]]:
        """Selecciona métodos de compresión apropiados"""
        methods = []
        
        if compression_ratio < 0.3:
            # Compresión ligera
            if layer_type in ['ffn', 'linear']:
                methods.append({'name': 'magnitude_pruning', 'strength': 0.3})
            elif layer_type == 'attention':
                methods.append({'name': 'head_pruning', 'strength': 0.2})
        
        elif compression_ratio < 0.6:
            # Compresión media
            if layer_type in ['ffn', 'linear']:
                methods.append({'name': 'magnitude_pruning', 'strength': 0.5})
                methods.append({'name': 'int8_quantization', 'strength': 0.5})
            elif layer_type == 'attention':
                methods.append({'name': 'head_pruning', 'strength': 0.4})
                methods.append({'name': 'low_rank_approximation', 'strength': 0.4})
        
        else:
            # Compresión agresiva
            if layer_type in ['ffn', 'linear']:
                methods.append({'name': 'structured_pruning', 'strength': 0.7})
                methods.append({'name': 'int4_quantization', 'strength': 0.8})
            elif layer_type == 'attention':
                methods.append({'name': 'attention_pruning', 'strength': 0.6})
                methods.append({'name': 'tensor_decomposition', 'strength': 0.7})
        
        return methods
    
    def _create_custom_strategy(self) -> Dict[str, Any]:
        """Crea estrategia personalizada basada en análisis"""
        strategy = {
            'name': 'custom_optimized',
            'description': 'Estrategia optimizada basada en análisis del modelo',
            'layer_configs': {},
            'preserve_layers': []
        }
        
        # Identificar capas a preservar
        for name, profile in self.layer_profiles.items():
            if profile.is_critical or profile.compression_potential < 0.1:
                strategy['preserve_layers'].append(name)
        
        # Configurar compresión adaptativa
        for name, profile in self.layer_profiles.items():
            if name not in strategy['preserve_layers']:
                # Compresión basada en potencial
                if profile.compression_potential > 0.7:
                    methods = [
                        {'name': 'aggressive_pruning', 'strength': 0.8},
                        {'name': 'int4_quantization', 'strength': 0.9}
                    ]
                elif profile.compression_potential > 0.4:
                    methods = [
                        {'name': 'balanced_pruning', 'strength': 0.5},
                        {'name': 'int8_quantization', 'strength': 0.6}
                    ]
                else:
                    methods = [
                        {'name': 'light_pruning', 'strength': 0.3}
                    ]
                
                strategy['layer_configs'][name] = {
                    'methods': methods,
                    'compression_ratio': profile.compression_potential * 0.8
                }
        
        return strategy
    
    def _get_recommended_strategy(self) -> str:
        """Determina estrategia recomendada basada en el modelo"""
        total_size = self.model_stats.get('total_size_mb', 0)
        
        # Basado en tamaño
        if total_size < 500:
            return 'conservative'
        elif total_size < 2000:
            return 'balanced'
        else:
            return 'aggressive'