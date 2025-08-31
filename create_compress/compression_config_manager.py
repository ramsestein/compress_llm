"""
Manejador principal de configuración de compresión - OPTIMIZADO
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from create_compress.compression_profiles import COMPRESSION_PROFILES

# Configurar logging
logger = logging.getLogger(__name__)

# Cache global para evitar recargas
_CONFIG_CACHE = {}
_PROFILE_CACHE = {}

class CompressionConfigManager:
    """Maneja la creación y gestión de configuraciones de compresión"""
    
    def __init__(self, model_path: str, output_dir: str = "./compression_analysis"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = self.model_path.name
        self.layer_types = defaultdict(list)
        self.compression_config = self._initialize_config()
        
        # Cache para evitar recálculos
        self._summary_cache = None
        self._layer_index_cache = {}
        
        # Lazy loading - no cargar la config del modelo hasta que sea necesaria
        self._model_config = None
        
    def _initialize_config(self) -> Dict[str, Any]:
        """Inicializa la estructura de configuración"""
        return {
            'metadata': {
                'model_name': self.model_name,
                'model_path': str(self.model_path),
                'created_date': datetime.now().isoformat(),
                'version': '1.0'
            },
            'global_settings': {},
            'layer_configs': {}
        }
    
    @property
    def config(self):
        """Lazy loading de la configuración del modelo"""
        if self._model_config is None:
            self._model_config = self._load_model_config()
        return self._model_config
    
    def _load_model_config(self) -> Optional[Any]:
        """Carga la configuración del modelo con cache"""
        cache_key = str(self.model_path)
        
        # Verificar cache
        if cache_key in _CONFIG_CACHE:
            logger.debug(f"Usando configuración cacheada para {self.model_name}")
            return _CONFIG_CACHE[cache_key]
        
        try:
            # Solo importar cuando sea necesario
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(self.model_path))
            _CONFIG_CACHE[cache_key] = config
            return config
        except Exception as e:
            logger.warning(f"⚠️ Error cargando configuración: {e}")
            return None
    
    def load_from_report(self) -> bool:
        """Carga información de capas desde el reporte de análisis - OPTIMIZADO"""
        report_path = self._find_report_path()
        
        if not report_path or not report_path.exists():
            self._print_missing_report_error()
            return False
        
        try:
            # Cargar reporte con optimizaciones
            logger.info(f"📊 Cargando reporte: {report_path.name}")
            
            # Usar lectura rápida para archivos grandes
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # Procesar capas en paralelo si hay muchas
            all_layers = report.get('complete_layer_analysis', {}).get('all_layers', {})
            self.total_layers = len(all_layers)
            
            if self.total_layers > 100:  # Umbral para procesamiento paralelo
                self._process_layers_parallel(all_layers)
            else:
                self._process_layers_sequential(all_layers)
            
            # Mostrar resumen
            self._show_layer_summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando reporte: {e}")
            return False
    
    def _find_report_path(self) -> Optional[Path]:
        """Encuentra el path del reporte con búsqueda optimizada"""
        base_name = self.model_name
        
        # Lista expandida de posibles nombres
        candidates = [
            # Formato esperado por compression_config
            self.output_dir / f"{base_name}_report_complete.json",
            self.output_dir / f"{base_name.replace('/', '_')}_report_complete.json",
            
            # Formatos de report_generator con timestamp
            *list(self.output_dir.glob(f"{base_name}_analysis_report_*.json")),
            *list(self.output_dir.glob(f"{base_name.replace('/', '_')}_analysis_report_*.json"))
        ]
        
        # Buscar el más reciente si hay varios
        existing_files = [p for p in candidates if isinstance(p, Path) and p.exists()]
        
        if existing_files:
            # Si hay múltiples, usar el más reciente
            return max(existing_files, key=lambda p: p.stat().st_mtime)
        
        # Si no encontramos ninguno, buscar con patrón más amplio
        json_files = list(self.output_dir.glob(f"*{base_name}*.json"))
        if json_files:
            return max(json_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def _process_layers_sequential(self, all_layers: Dict[str, Any]):
        """Procesa capas secuencialmente para modelos pequeños"""
        for layer_name, layer_info in all_layers.items():
            layer_type = layer_info.get('type', 'other')
            self.layer_types[layer_type].append({
                'name': layer_name,
                'size_mb': layer_info.get('size_mb', 0),
                'parameters': layer_info.get('parameters', 0),
                'position': layer_info.get('relative_position', 0),
                'layer_index': layer_info.get('layer_index', 0)
            })
    
    def _process_layers_parallel(self, all_layers: Dict[str, Any]):
        """Procesa capas en paralelo para modelos grandes"""
        def process_batch(items):
            result = defaultdict(list)
            for layer_name, layer_info in items:
                layer_type = layer_info.get('type', 'other')
                result[layer_type].append({
                    'name': layer_name,
                    'size_mb': layer_info.get('size_mb', 0),
                    'parameters': layer_info.get('parameters', 0),
                    'position': layer_info.get('relative_position', 0),
                    'layer_index': layer_info.get('layer_index', 0)
                })
            return result
        
        # Dividir en lotes para procesamiento paralelo
        items = list(all_layers.items())
        n_cores = mp.cpu_count()
        batch_size = max(1, len(items) // (n_cores * 2))
        
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(process_batch, batches))
        
        # Combinar resultados
        for result in results:
            for layer_type, layers in result.items():
                self.layer_types[layer_type].extend(layers)
    
    def _show_layer_summary(self):
        """Muestra resumen de capas"""
        print(f"\n📋 Tipos de capas encontrados:")
        total_size = 0
        
        for layer_type, layers in sorted(self.layer_types.items()):
            type_size = sum(l['size_mb'] for l in layers)
            total_size += type_size
            print(f"  - {layer_type}: {len(layers)} capas, {type_size:.1f} MB")
        
        print(f"\n  Total: {total_size:.1f} MB en {self.total_layers} capas")

    def get_layer_types_list(self) -> List[str]:
        """Obtiene la lista de tipos de capas disponibles"""
        return list(self.layer_types.keys())
    
    def get_layers_by_type(self, layer_type: str) -> List[Dict[str, Any]]:
        """Obtiene todas las capas de un tipo específico"""
        return self.layer_types.get(layer_type, [])
    
    def _print_missing_report_error(self):
        """Imprime error cuando no se encuentra el reporte"""
        print(f"\n❌ No se encontró reporte de análisis previo.")
        print(f"   Buscado en: {self.output_dir}")
        print("\n📝 Primero debes ejecutar el análisis del modelo:")
        print(f"   python analyze_model.py {self.model_name}")
        print("\nEsto generará el reporte necesario para crear la configuración de compresión.")
    
    def set_compression_profile(self, profile_name: str):
        """Configura un perfil de compresión con cache"""
        # Cache de perfiles
        if profile_name not in _PROFILE_CACHE:
            # Lazy import
            _PROFILE_CACHE.update(COMPRESSION_PROFILES)
        
        profile = _PROFILE_CACHE.get(profile_name)
        if not profile:
            raise ValueError(f"Perfil no encontrado: {profile_name}")
        
        print(f"\n✅ Aplicando perfil: {profile['description']}")
        
        self.compression_config['global_settings'] = profile

        # Copiar config de capas del perfil seleccionado
        layer_configs = profile.get('layer_configs', {}).copy()

        # Tipos estándar que deben existir incluso si no aparecen en el
        # análisis del modelo. Esto garantiza que capas como las de
        # normalización tengan una configuración explícita y puedan ser
        # modificadas posteriormente por el usuario.
        standard_types = {
            'embedding', 'attention', 'ffn', 'linear', 'normalization',
            'output', 'conv', 'other', 'skip'
        }

        # Asegurar que todos los tipos detectados o estándar tengan al menos
        # una configuración "sin compresión" para evitar errores durante la
        # aplicación.
        for layer_type in set(self.layer_types.keys()) | standard_types:
            if layer_type not in layer_configs:
                layer_configs[layer_type] = {
                    'methods': [{'name': 'none', 'strength': 0.0}],
                    'total_compression_ratio': 0.0
                }

        self.compression_config['layer_configs'] = layer_configs

        # Invalidar cache de resumen
        self._summary_cache = None
    
    def set_custom_layer_config(self, layer_type: str, methods: List[Dict[str, Any]], 
                               total_compression_ratio: float):
        """Configura compresión personalizada para un tipo de capa"""
        self.compression_config['layer_configs'][layer_type] = {
            'methods': methods,
            'total_compression_ratio': total_compression_ratio
        }
        
        # Invalidar cache
        self._summary_cache = None
    
    def add_position_modifiers(self):
        """Agrega modificadores de compresión por posición"""
        self.compression_config['position_modifiers'] = {
            'early_layers': {
                'range': [0.0, 0.3],
                'compression_multiplier': 0.7,
                'description': 'Capas iniciales (importantes para features)'
            },
            'middle_layers': {
                'range': [0.3, 0.7],
                'compression_multiplier': 1.0,
                'description': 'Capas intermedias'
            },
            'late_layers': {
                'range': [0.7, 1.0],
                'compression_multiplier': 1.3,
                'description': 'Capas finales (menos críticas)'
            }
        }
    
    def add_preserved_layers(self, layer_names: List[str]):
        """Marca capas específicas para preservar sin compresión"""
        if 'preserved_layers' not in self.compression_config:
            self.compression_config['preserved_layers'] = []
        
        # Usar set para evitar duplicados
        preserved_set = set(self.compression_config['preserved_layers'])
        preserved_set.update(layer_names)
        self.compression_config['preserved_layers'] = sorted(list(preserved_set))
    
    def set_validation_settings(self):
        """Establece configuración de validación"""
        self.compression_config['validation'] = {
            'min_performance_threshold': 0.9,
            'max_compression_per_layer': 0.8,
            'require_gradual_compression': True,
            'validation_samples': 100
        }
    
    def set_final_layers_config(self, num_layers: int, config: Dict[str, Any]):
        """Configura las últimas N capas con configuración especial"""
        self.compression_config['final_layers_count'] = num_layers
        self.compression_config['final_layers_config'] = config
        
        # Invalidar cache
        self._summary_cache = None
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calcula resumen de la configuración con cache"""
        # Usar cache si está disponible
        if self._summary_cache is not None:
            return self._summary_cache
        
        summary = {
            'original_size_mb': 0,
            'compressed_size_mb': 0,
            'compression_ratio': 0,
            'by_layer_type': {}
        }
        
        # Calcular por tipo de capa
        for layer_type, layers in self.layer_types.items():
            layer_config = self.compression_config['layer_configs'].get(layer_type, {})
            compression_ratio = layer_config.get('total_compression_ratio', 0)
            
            type_size = sum(l['size_mb'] for l in layers)
            compressed_size = type_size * (1 - compression_ratio)
            
            summary['by_layer_type'][layer_type] = {
                'num_layers': len(layers),
                'original_size_mb': round(type_size, 2),
                'compressed_size_mb': round(compressed_size, 2),
                'compression_ratio': compression_ratio
            }
            
            summary['original_size_mb'] += type_size
            summary['compressed_size_mb'] += compressed_size
        
        # Calcular ratio total
        if summary['original_size_mb'] > 0:
            summary['compression_ratio'] = (
                summary['original_size_mb'] - summary['compressed_size_mb']
            ) / summary['original_size_mb']
        
        # Cachear resultado
        self._summary_cache = summary
        return summary
    
    def load_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Carga una configuración desde un archivo"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Configuración cargada desde: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            return None

    def save_config(self, config: Dict[str, Any], config_path: Path) -> bool:
        """Guarda una configuración en un archivo"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Configuración guardada en: {config_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando configuración: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida una configuración de compresión"""
        errors = []
        
        # Validar estructura básica
        if 'metadata' not in config:
            errors.append("Falta sección 'metadata'")
        
        if 'global_settings' not in config:
            errors.append("Falta sección 'global_settings'")
        
        # Validar metadata
        metadata = config.get('metadata', {})
        if 'model_name' not in metadata:
            errors.append("Falta 'model_name' en metadata")
        
        # Validar configuración global
        global_settings = config.get('global_settings', {})
        if 'target_compression' in global_settings:
            target = global_settings['target_compression']
            if not isinstance(target, (int, float)) or target < 0 or target > 1:
                errors.append("'target_compression' debe ser un número entre 0 y 1")
        
        # Validar configuraciones de capas
        layer_configs = config.get('layer_configs', {})
        for layer_name, layer_config in layer_configs.items():
            if 'methods' not in layer_config:
                errors.append(f"Falta 'methods' en configuración de capa '{layer_name}'")
            else:
                methods = layer_config['methods']
                if not isinstance(methods, list):
                    errors.append(f"'methods' debe ser una lista en capa '{layer_name}'")
                else:
                    for i, method in enumerate(methods):
                        if not isinstance(method, dict):
                            errors.append(f"Método {i} en capa '{layer_name}' debe ser un diccionario")
                        elif 'name' not in method:
                            errors.append(f"Falta 'name' en método {i} de capa '{layer_name}'")
                        elif 'strength' not in method:
                            errors.append(f"Falta 'strength' en método {i} de capa '{layer_name}'")
                        else:
                            strength = method['strength']
                            if not isinstance(strength, (int, float)) or strength < 0 or strength > 1:
                                errors.append(f"'strength' debe ser un número entre 0 y 1 en método {i} de capa '{layer_name}'")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("✅ Configuración válida")
        else:
            logger.warning(f"⚠️ Configuración inválida: {len(errors)} errores")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors

    def save_configuration(self, filename: Optional[str] = None):
        """Guarda la configuración en JSON"""
        if filename is None:
            filename = f"{self.model_name}_compression_config.json"
        
        output_path = self.output_dir / filename
        
        # Agregar resumen antes de guardar
        self.compression_config['summary'] = self.calculate_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.compression_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Configuración guardada en: {output_path}")
        return output_path