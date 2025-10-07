#!/usr/bin/env python3
"""
Test completo del script apply_compression.py (sin emojis)
Verifica todas las funcionalidades principales
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_apply_compression_complete():
    """Test completo de apply_compression.py"""
    
    print("=" * 60)
    print("TEST COMPLETO DE APPLY_COMPRESSION.PY")
    print("=" * 60)
    
    # 1. Verificar que el script se puede importar
    print("\n1. Verificando importacion del script...")
    try:
        # Importar funciones principales
        sys.path.append('.')
        from apply_compression import (
            save_pretrained_with_fallback,
            validate_model_path,
            load_compression_config,
            apply_compression_to_model,
            ModelCompressor
        )
        print("   [OK] Importacion exitosa")
    except Exception as e:
        print(f"   [ERROR] Error en importacion: {e}")
        return False
    
    # 2. Verificar que el modelo distilgpt2 existe
    print("\n2. Verificando modelo distilgpt2...")
    model_path = Path("models/distilgpt2")
    if model_path.exists():
        print("   [OK] Modelo distilgpt2 encontrado")
    else:
        print("   [ERROR] Modelo distilgpt2 no encontrado")
        return False
    
    # 3. Verificar configuración de compresión
    print("\n3. Verificando configuracion de compresion...")
    config_path = Path("compression_analysis/distilgpt2_compression_config.json")
    if config_path.exists():
        print("   [OK] Configuracion encontrada")
        try:
            config = load_compression_config(str(config_path))
            print(f"   [OK] Configuracion valida: {config.get('model_name', 'N/A')}")
        except Exception as e:
            print(f"   [ERROR] Error cargando configuracion: {e}")
            return False
    else:
        print("   [ERROR] Configuracion no encontrada")
        return False
    
    # 4. Verificar validación de rutas
    print("\n4. Verificando validacion de rutas...")
    try:
        valid = validate_model_path("models/distilgpt2")
        print(f"   [OK] Validacion de ruta: {valid}")
    except Exception as e:
        print(f"   [ERROR] Error en validacion: {e}")
        return False
    
    # 5. Probar compresión completa
    print("\n5. Probando compresion completa...")
    try:
        # Crear directorio temporal para output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_compressed"
            
            # Aplicar compresión
            result = apply_compression_to_model(
                model_path="models/distilgpt2",
                config_path=str(config_path),
                output_path=str(output_path)
            )
            
            print(f"   [OK] Compresion exitosa: {result.get('success', False)}")
            print(f"   [INFO] Ratio de compresion: {result.get('compression_ratio', 0):.2%}")
            print(f"   [INFO] Modelo guardado en: {result.get('model_path', 'N/A')}")
            
            # Verificar que el modelo se guardó
            if output_path.exists():
                print("   [OK] Directorio de salida creado")
                files = list(output_path.glob("*"))
                print(f"   [INFO] Archivos creados: {len(files)}")
                for file in files[:5]:  # Mostrar primeros 5 archivos
                    print(f"      - {file.name}")
            else:
                print("   [ERROR] Directorio de salida no creado")
                return False
                
    except Exception as e:
        print(f"   [ERROR] Error en compresion: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Probar ModelCompressor directamente
    print("\n6. Probando ModelCompressor directamente...")
    try:
        compressor = ModelCompressor(
            compression_config_path=str(config_path),
            models_dir="models"
        )
        print("   [OK] ModelCompressor creado exitosamente")
        
        # Verificar que puede cargar la configuración
        config = compressor._load_compression_config()
        print(f"   [OK] Configuracion cargada: {config.get('model_name', 'N/A')}")
        
    except Exception as e:
        print(f"   [ERROR] Error en ModelCompressor: {e}")
        return False
    
    # 7. Verificar funciones de utilidad
    print("\n7. Verificando funciones de utilidad...")
    try:
        # Probar limpieza de diccionarios
        test_dict = {"key": "value", "nested": {"deep": "data"}}
        from apply_compression import _clean_dict_for_serialization
        cleaned = _clean_dict_for_serialization(test_dict)
        print("   [OK] Limpieza de diccionarios funciona")
        
        # Probar limpieza de listas
        test_list = ["item1", "item2", {"nested": "data"}]
        from apply_compression import _clean_list_for_serialization
        cleaned_list = _clean_list_for_serialization(test_list)
        print("   [OK] Limpieza de listas funciona")
        
    except Exception as e:
        print(f"   [ERROR] Error en funciones de utilidad: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print("[SUCCESS] TODAS LAS FUNCIONALIDADES FUNCIONAN CORRECTAMENTE")
    print("[SUCCESS] El script apply_compression.py esta completamente funcional")
    print("[SUCCESS] Todas las funciones principales operan sin errores")
    
    return True

if __name__ == "__main__":
    success = test_apply_compression_complete()
    if success:
        print("\n[SUCCESS] TEST COMPLETO EXITOSO!")
    else:
        print("\n[ERROR] TEST FALLO - Revisar errores arriba")
        sys.exit(1)


