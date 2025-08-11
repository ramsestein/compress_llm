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
        
        # Default: sin compresión
        logger.warning(f"⚠️ No hay configuración para tipo '{layer_type}', preservando: {layer_name}")
        return {
            'methods': [{'name': 'none', 'strength': 0.0}],
            'total_compression_ratio': 0.0
        }
    
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
                            
                            # Reemplazar módulo si fue modificado
                            if compressed_module is not module:
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
            
            # 5. Guardar modelo comprimido
            logger.info("\n💾 Guardando modelo comprimido...")
            model.save_pretrained(self.output_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(self.output_path)
            
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
        """Determina el tipo de una capa"""
        name_lower = name.lower()
        module_type = type(module).__name__.lower()
        
        # Embeddings
        if 'embed' in name_lower or 'embed' in module_type:
            return 'embedding'
        
        # Attention
        if any(x in name_lower for x in ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'attention'
        
        # FFN/MLP
        if any(x in name_lower for x in ['mlp', 'ffn', 'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']):
            return 'ffn'
        
        # Normalization
        if any(x in name_lower for x in ['norm', 'layernorm', 'ln']):
            return 'normalization'
        
        # Output/LM Head
        if any(x in name_lower for x in ['lm_head', 'output', 'classifier']):
            return 'output'
        
        return 'other'
    
    def _is_compressible_layer(self, module: nn.Module) -> bool:
        """Determina si una capa es comprimible"""
        # Solo comprimir capas con parámetros significativos
        if not hasattr(module, 'parameters'):
            return False
        
        num_params = sum(p.numel() for p in module.parameters())
        
        # Umbral mínimo de parámetros
        return num_params > 10000
    
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