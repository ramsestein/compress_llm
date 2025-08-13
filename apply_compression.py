#!/usr/bin/env python3
"""
Script principal para aplicar compresión a modelos según configuración JSON
Versión con soporte completo para todos los métodos de compresión
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel
)
from tqdm import tqdm
import logging
from datetime import datetime
import gc

# Importar el motor de compresión
from create_compress.compression_engine import CompressionEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelCompressor:
    """Gestor principal de compresión de modelos"""
    
    def __init__(self, compression_config_path: str, models_dir: str = "./models", 
                 output_suffix: str = "_compressed"):
        self.config_path = Path(compression_config_path)
        self.models_dir = Path(models_dir)
        self.output_suffix = output_suffix
        
        # Cargar configuración
        self.compression_config = self._load_compression_config()
        self.model_name = self.compression_config['metadata']['model_name']
        self.model_path = self.models_dir / self.model_name
        self.output_path = self.models_dir / f"{self.model_name}{self.output_suffix}"
        
        # Motor de compresión
        self.engine = CompressionEngine()
        
        # Estadísticas
        self.stats = {
            'original_size_mb': 0,
            'compressed_size_mb': 0,
            'layers_compressed': 0,
            'layers_preserved': 0,
            'compression_time_seconds': 0,
            'final_layers_compressed': 0,
            'methods_used': set()
        }
    
    def _load_compression_config(self) -> Dict[str, Any]:
        """Carga la configuración de compresión"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"No se encontró configuración: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"✅ Configuración cargada: {self.config_path.name}")
        logger.info(f"📦 Modelo objetivo: {config['metadata']['model_name']}")
        logger.info(f"🎯 Perfil: {config['global_settings']['profile']}")
        logger.info(f"📊 Compresión objetivo: {config['global_settings']['target_compression']*100:.1f}%")
        
        # Verificar si hay configuración de capas finales
        if config.get('final_layers_config'):
            final_count = config.get('final_layers_count', 0)
            logger.info(f"🎯 Configuración especial para las últimas {final_count} capas")
        
        return config
    
    def _get_layer_config(self, layer_name: str, layer_type: str, 
                         relative_position: float, layer_index: int, 
                         total_layers: int) -> Dict[str, Any]:
        """Obtiene configuración de compresión para una capa específica"""
        
        # Verificar si es una capa final con configuración especial
        final_layers_config = self.compression_config.get('final_layers_config')
        final_layers_count = self.compression_config.get('final_layers_count', 0)
        
        if final_layers_config and final_layers_count > 0:
            # Calcular si esta capa está en las capas finales
            layers_from_end = total_layers - layer_index
            if layers_from_end <= final_layers_count:
                logger.debug(f"🎯 Aplicando configuración de capas finales a: {layer_name} (capa {layers_from_end} desde el final)")
                self.stats['final_layers_compressed'] += 1
                return final_layers_config
        
        # Verificar si la capa está en la lista de preservadas
        preserved_layers = self.compression_config.get('preserved_layers', [])
        if layer_name in preserved_layers:
            logger.info(f"🛡️ Preservando capa sin cambios: {layer_name}")
            self.stats['layers_preserved'] += 1
            return {
                'methods': [{'name': 'none', 'strength': 0.0}],
                'total_compression_ratio': 0.0
            }
        
        # Usar configuración por tipo de capa
        layer_configs = self.compression_config.get('layer_configs', {})
        if layer_type in layer_configs:
            return layer_configs[layer_type]
        # Si llegamos aquí, la configuración está incompleta
        raise KeyError(
            f"No hay configuración para el tipo de capa '{layer_type}' (capa: {layer_name})"
        )
    
    def compress_model(self):
        """Ejecuta la compresión del modelo"""
        start_time = datetime.now()
        
        # Verificar que el modelo existe
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 INICIANDO COMPRESIÓN DE MODELO")
        logger.info(f"{'='*60}")
        logger.info(f"📦 Modelo: {self.model_name}")
        logger.info(f"📁 Entrada: {self.model_path}")
        logger.info(f"📁 Salida: {self.output_path}")
        logger.info(f"{'='*60}\n")
        
        # Crear directorio de salida
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Cargar modelo y configuración
            logger.info("📥 Cargando modelo...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Cargar configuración primero para obtener arquitectura
            config = AutoConfig.from_pretrained(self.model_path)
            model_type = config.model_type
            
            # Cargar tokenizer si existe
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception:
                tokenizer = None
                logger.warning("Tokenizer no encontrado, continuando sin él")
            
            # Cargar modelo con configuración de memoria optimizada
            logger.info(f"🖥️ Dispositivo: {device}")
            logger.info(f"🏗️ Arquitectura: {model_type}")
            
            if device.type == "cuda":
                # Cargar en FP16 para ahorrar memoria
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                model = model.to(device)
            
            # 2. Calcular tamaño original
            self.stats['original_size_mb'] = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 * 1024)
            logger.info(f"📊 Tamaño original: {self.stats['original_size_mb']:.1f} MB")
            
            # 3. Aplicar compresión capa por capa
            logger.info("\n🔧 Aplicando compresión...")
            
            # Obtener todas las capas con nombres
            named_modules = list(model.named_modules())
            total_layers = len(named_modules)
            
            with tqdm(total=total_layers, desc="Comprimiendo capas") as pbar:
                for layer_index, (name, module) in enumerate(named_modules):
                    # Determinar tipo de capa
                    layer_type = self._get_layer_type(name, module)
                    
                    # Saltar si no es una capa comprimible
                    if not self._is_compressible_layer(module):
                        pbar.update(1)
                        continue
                    
                    # Calcular posición relativa
                    relative_position = layer_index / total_layers if total_layers > 0 else 0
                    
                    # Obtener configuración para esta capa
                    layer_config = self._get_layer_config(
                        name, layer_type, relative_position, 
                        layer_index, total_layers
                    )
                    
                    # Aplicar métodos de compresión
                    for method_config in layer_config['methods']:
                        method_name = method_config['name']
                        strength = method_config['strength']
                        
                        if method_name != 'none' and strength > 0:
                            logger.debug(f"  Aplicando {method_name} ({strength*100:.0f}%) a {name}")
                            
                            # Aplicar método usando el motor
                            compressed_module = self.engine.apply_method(
                                module, method_name, strength, layer_config
                            )
                            
                            self._replace_module(model, name, compressed_module)
                            
                            self.stats['methods_used'].add(method_name)
                    
                    # Actualizar estadísticas
                    if layer_config['total_compression_ratio'] > 0:
                        self.stats['layers_compressed'] += 1
                    
                    pbar.update(1)
            
            # 4. Optimizaciones post-compresión
            logger.info("\n⚡ Aplicando optimizaciones finales...")
            
            # Limpiar buffers no usados
            self._cleanup_model(model)

            logger.info("\n💾 Guardando modelo comprimido...")
            # 5. Guardar modelo comprimido
            # Guardar el modelo, reintentando si se alcanza el límite de
            # recursión.  En algunos modelos con muchas capas o estructuras
            # modificadas por la compresión, `save_pretrained` puede requerir
            # un límite de recursión mayor al predeterminado de Python.  Se
            # incrementa progresivamente hasta tres veces para evitar que la
            # ejecución termine con un `RecursionError`.
            import sys
            for attempt in range(3):
                try:
                    model.save_pretrained(self.output_path)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(self.output_path)
                    break
                except RecursionError:
                    new_limit = sys.getrecursionlimit() * 2
                    logger.error(
                        f"RecursionError al guardar (intento {attempt + 1}), "
                        f"aumentando límite a {new_limit}"
                    )
                    sys.setrecursionlimit(new_limit)
            else:
                # Si después de varios intentos sigue fallando, propagar el
                # error para que el usuario tenga visibilidad.
                raise
            
            # 6. Copiar archivos adicionales
            self._copy_additional_files()
            
            # 7. Guardar estadísticas y configuración
            self._save_compression_info()
            
            # Calcular tiempo total
            self.stats['compression_time_seconds'] = (datetime.now() - start_time).total_seconds()
            
            # 8. Mostrar resumen
            self._print_summary()
            
            # Limpiar memoria
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Error durante la compresión: {str(e)}")
            raise
    
    def _copy_additional_files(self):
        """Copia archivos adicionales del modelo original"""
        files_to_copy = [
            'config.json',
            'generation_config.json',
            'special_tokens_map.json',
            'tokenizer_config.json',
            'tokenizer.json',
            'vocab.json',
            'merges.txt',
            'added_tokens.json',
            'preprocessor_config.json'
        ]
        
        for filename in files_to_copy:
            src = self.model_path / filename
            dst = self.output_path / filename
            
            if src.exists():
                shutil.copy2(src, dst)
                logger.debug(f"📄 Copiado: {filename}")
    
    def _save_compression_info(self):
        """Guarda información sobre la compresión aplicada"""
        # Convertir set a lista para JSON
        self.stats['methods_used'] = list(self.stats['methods_used'])
        
        info = {
            'compression_date': datetime.now().isoformat(),
            'original_model': str(self.model_path),
            'compression_config': self.compression_config,
            'statistics': self.stats,
            'notes': []
        }
        
        # Agregar nota sobre capas finales
        if self.stats['final_layers_compressed'] > 0:
            info['notes'].append(
                f"Se aplicó configuración especial a las últimas {self.stats['final_layers_compressed']} capas"
            )
        
        # Guardar
        info_path = self.output_path / "compression_metadata.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        # También copiar la configuración original
        shutil.copy2(self.config_path, self.output_path / "compression_config.json")
    
    def _print_summary(self):
        """Imprime resumen de la compresión"""
        # Calcular tamaño final
        compressed_size = sum(
            os.path.getsize(os.path.join(root, file))
            for root, _, files in os.walk(self.output_path)
            for file in files
            if file.endswith(('.bin', '.safetensors', '.pt', '.pth'))
        ) / (1024 * 1024)
        
        self.stats['compressed_size_mb'] = compressed_size
        compression_ratio = 1 - (compressed_size / self.stats['original_size_mb'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ COMPRESIÓN COMPLETADA")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Estadísticas:")
        logger.info(f"   • Tamaño original:     {self.stats['original_size_mb']:.1f} MB")
        logger.info(f"   • Tamaño comprimido:   {compressed_size:.1f} MB")
        logger.info(f"   • Reducción:           {compression_ratio*100:.1f}%")
        logger.info(f"   • Factor:              {self.stats['original_size_mb']/compressed_size:.1f}x")
        logger.info(f"\n📈 Capas procesadas:")
        logger.info(f"   • Comprimidas:         {self.stats['layers_compressed']}")
        logger.info(f"   • Preservadas:         {self.stats['layers_preserved']}")
        logger.info(f"   • Capas finales:       {self.stats['final_layers_compressed']}")
        logger.info(f"\n🔧 Métodos utilizados:   {', '.join(self.stats['methods_used'])}")
        logger.info(f"⏱️ Tiempo:               {self.stats['compression_time_seconds']:.1f} segundos")
        logger.info(f"\n💾 Modelo guardado en:   {self.output_path}")
        logger.info(f"{'='*60}")
    
    def _get_layer_type(self, name: str, module: nn.Module) -> str:
        """Determina el tipo de una capa de forma genérica"""
        name_lower = name.lower()
        module_type = type(module).__name__.lower()
        
        # 1. EMBEDDINGS - Patrones universales
        embedding_patterns = ['embed', 'emb', 'wte', 'wpe', 'position', 'token']
        if any(pattern in name_lower for pattern in embedding_patterns):
            return 'embedding'
        if isinstance(module, nn.Embedding):
            return 'embedding'
        
        # 2. NORMALIZATION - Por tipo de módulo
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            return 'normalization'
        norm_patterns = ['norm', 'ln', 'layernorm', 'batchnorm', 'groupnorm']
        if any(pattern in name_lower for pattern in norm_patterns):
            return 'normalization'
        
        # 3. OUTPUT/HEAD - Patrones comunes
        output_patterns = ['head', 'output', 'classifier', 'lm_head', 'cls', 'prediction', 'logits', 'score']
        if any(pattern in name_lower for pattern in output_patterns):
            return 'output'
        
        # 4. ATTENTION - Patrones multi-arquitectura
        attention_patterns = [
            'attention', 'attn', 'self_attn', 'cross_attn',
            'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Común en LLaMA, GPT
            'query', 'key', 'value',  # BERT style
            'c_attn', 'c_proj',  # GPT-2 style
            'qkv', 'out_proj'  # Algunos modelos combinan QKV
        ]
        if any(pattern in name_lower for pattern in attention_patterns):
            return 'attention'
        
        # Detectar por tipo de módulo (MultiheadAttention, etc.)
        if 'attention' in module_type or 'multihead' in module_type:
            return 'attention'
        
        # 5. FFN/MLP - Patrones universales
        ffn_patterns = [
            'mlp', 'ffn', 'feed_forward', 'feedforward',
            'fc', 'dense',  # Fully connected
            'w1', 'w2', 'w3',  # Algunos modelos usan esta nomenclatura
            'gate_proj', 'up_proj', 'down_proj',  # LLaMA style
            'c_fc', 'c_proj',  # GPT style
            'intermediate', 'output.dense'  # BERT style
        ]
        if any(pattern in name_lower for pattern in ffn_patterns):
            return 'ffn'
        
        # 6. ANÁLISIS ESTRUCTURAL para Linear layers
        if isinstance(module, nn.Linear):
            # Analizar dimensiones para inferir tipo
            in_features = module.in_features
            out_features = module.out_features
            
            # Si está dentro de un bloque transformer (heurística)
            if '.h.' in name_lower or '.layer.' in name_lower or '.block.' in name_lower:
                # Buscar pistas en el nombre del padre
                parent_parts = name_lower.split('.')
                for i, part in enumerate(parent_parts):
                    # Si el Linear está después de algo que suena a attention
                    if i > 0 and any(attn in parent_parts[i-1] for attn in ['attn', 'attention']):
                        return 'attention'
                    # Si está después de algo que suena a MLP/FFN
                    if i > 0 and any(ffn in parent_parts[i-1] for ffn in ['mlp', 'ffn', 'feed']):
                        return 'ffn'
                
                # Heurística por tamaño: FFN suele tener expansión 4x
                if out_features > in_features * 3 or in_features > out_features * 3:
                    return 'ffn'
                
                # Si las dimensiones son iguales, podría ser attention
                if in_features == out_features:
                    return 'attention'
            
            return 'linear'  # Mejor que 'other' para capas Linear
        
        # 7. CONVOLUCIONAL (por si acaso)
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return 'conv'
        
        # 8. DROPOUT y otros (no se comprimen)
        if isinstance(module, (nn.Dropout, nn.Identity)):
            return 'skip'  # Nueva categoría para capas que no se tocan
        
        # 9. Si no podemos determinar, intentar por estructura
        # Verificar si tiene sub-módulos que den pistas
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):  # Solo para contenedores específicos
            child_modules = list(module.named_children())
            if child_modules and len(name.split('.')) < 10:  # Limitar profundidad de recursión
                # Analizar hijos para inferir tipo del padre
                child_types = set()
                for child_name, child_module in child_modules[:3]:  # Solo primeros 3 hijos
                    # Evitar recursión analizando solo el tipo del módulo hijo directamente
                    if isinstance(child_module, nn.Linear):
                        child_types.add('linear')
                    elif isinstance(child_module, nn.LayerNorm):
                        child_types.add('normalization')
                    elif 'attention' in type(child_module).__name__.lower():
                        child_types.add('attention')
                    elif 'mlp' in type(child_module).__name__.lower():
                        child_types.add('ffn')
                
                # Si todos los hijos son del mismo tipo, el padre probablemente es ese tipo
                if len(child_types) == 1:
                    return child_types.pop()
        
        return 'other'
    
    def _is_compressible_layer(self, module: nn.Module) -> bool:
        """Determina si una capa es comprimible"""
        # Excluir tipos que nunca se comprimen
        if isinstance(module, (nn.Dropout, nn.Identity)):
            return False
        
        # Solo comprimir capas con parámetros significativos
        if not hasattr(module, 'parameters'):
            return False
        
        num_params = sum(p.numel() for p in module.parameters())
        
        # Umbral mínimo de parámetros (ajustable)
        min_params = 1000  # Reducido para capas más pequeñas
        
        return num_params > min_params
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Reemplaza un módulo en el modelo"""
        parts = module_name.split('.')
        parent = model
        
        # Navegar hasta el padre
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Reemplazar
        setattr(parent, parts[-1], new_module)
    
    def _cleanup_model(self, model: nn.Module):
        """Limpia buffers y optimiza el modelo"""
        # Eliminar buffers no esenciales
        for name, module in model.named_modules():
            # Limpiar cachés de atención si existen
            if hasattr(module, 'attention_cache'):
                delattr(module, 'attention_cache')
            
            # Compactar pesos si es posible
            for param_name, param in module.named_parameters():
                if param.grad is not None:
                    param.grad = None


def main():
    parser = argparse.ArgumentParser(
        description='Aplica compresión a un modelo según configuración JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Comprimir usando configuración generada
  python apply_compression.py llama-7b
  
  # Especificar archivo de configuración
  python apply_compression.py --config ./configs/mi_config.json
  
  # Usar sufijo personalizado
  python apply_compression.py llama-7b --suffix _optimized
        """
    )
    
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        help='Nombre del modelo a comprimir'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Ruta al archivo de configuración (default: busca en compression_analysis/)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='Directorio de modelos (default: ./models)'
    )
    
    parser.add_argument(
        '--suffix',
        type=str,
        default='_compressed',
        help='Sufijo para el modelo comprimido (default: _compressed)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Sobrescribir si el modelo comprimido ya existe'
    )
    
    args = parser.parse_args()
    
    # Determinar archivo de configuración
    if args.config:
        config_path = args.config
    elif args.model:
        # Buscar en directorio por defecto
        config_path = f"./compression_analysis/{args.model}_compression_config.json"
    else:
        parser.error("Debes especificar un modelo o un archivo de configuración")
    
    # Verificar que existe
    if not Path(config_path).exists():
        logger.error(f"❌ No se encontró archivo de configuración: {config_path}")
        logger.error("   Primero ejecuta: python create_compression_config.py <modelo>")
        sys.exit(1)
    
    # Verificar si ya existe el modelo comprimido
    models_dir = Path(args.models_dir)
    with open(config_path, 'r') as f:
        model_name = json.load(f)['metadata']['model_name']
    
    output_path = models_dir / f"{model_name}{args.suffix}"
    if output_path.exists() and not args.force:
        logger.error(f"❌ El modelo comprimido ya existe: {output_path}")
        logger.error("   Usa --force para sobrescribir")
        sys.exit(1)
    
    try:
        # Crear compresor y ejecutar
        compressor = ModelCompressor(config_path, args.models_dir, args.suffix)
        compressor.compress_model()
        
        # Sugerir próximos pasos
        logger.info("\n📝 Próximos pasos:")
        logger.info(f"1. Verificar el modelo:")
        logger.info(f"   python verify_compression.py {model_name}")
        logger.info(f"\n2. Probar el modelo:")
        logger.info(f"   python ollama_compact_server.py --model {model_name}{args.suffix}")
        logger.info(f"\n3. Fine-tuning (si es necesario):")
        logger.info(f"   python finetune_lora.py")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()