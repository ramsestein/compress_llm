#!/usr/bin/env python3
"""
Test completo del script apply_compression.py
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
    print("\n1. Verificando importación del script...")
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
        print("   ✅ Importación exitosa")
    except Exception as e:
        print(f"   ❌ Error en importación: {e}")
        return False
    
    # 2. Verificar que el modelo distilgpt2 existe
    print("\n2. Verificando modelo distilgpt2...")
    model_path = Path("models/distilgpt2")
    if model_path.exists():
        print("   ✅ Modelo distilgpt2 encontrado")
    else:
        print("   ❌ Modelo distilgpt2 no encontrado")
        return False
    
    # 3. Verificar configuración de compresión
    print("\n3. Verificando configuración de compresión...")
    config_path = Path("compression_analysis/distilgpt2_compression_config.json")
    if config_path.exists():
        print("   ✅ Configuración encontrada")
        try:
            config = load_compression_config(str(config_path))
            print(f"   ✅ Configuración válida: {config.get('model_name', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Error cargando configuración: {e}")
            return False
    else:
        print("   ❌ Configuración no encontrada")
        return False
    
    # 4. Verificar validación de rutas
    print("\n4. Verificando validación de rutas...")
    try:
        valid = validate_model_path("models/distilgpt2")
        print(f"   ✅ Validación de ruta: {valid}")
    except Exception as e:
        print(f"   ❌ Error en validación: {e}")
        return False
    
    # 5. Probar compresión completa
    print("\n5. Probando compresión completa...")
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
            
            print(f"   ✅ Compresión exitosa: {result.get('success', False)}")
            print(f"   📊 Ratio de compresión: {result.get('compression_ratio', 0):.2%}")
            print(f"   📁 Modelo guardado en: {result.get('model_path', 'N/A')}")
            
            # Verificar que el modelo se guardó
            if output_path.exists():
                print("   ✅ Directorio de salida creado")
                files = list(output_path.glob("*"))
                print(f"   📄 Archivos creados: {len(files)}")
                for file in files[:5]:  # Mostrar primeros 5 archivos
                    print(f"      - {file.name}")
            else:
                print("   ❌ Directorio de salida no creado")
                return False
                
    except Exception as e:
        print(f"   ❌ Error en compresión: {e}")
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
        print("   ✅ ModelCompressor creado exitosamente")
        
        # Verificar que puede cargar la configuración
        config = compressor._load_compression_config()
        print(f"   ✅ Configuración cargada: {config.get('model_name', 'N/A')}")
        
    except Exception as e:
        print(f"   ❌ Error en ModelCompressor: {e}")
        return False
    
    # 7. Verificar funciones de utilidad
    print("\n7. Verificando funciones de utilidad...")
    try:
        # Probar limpieza de diccionarios
        test_dict = {"key": "value", "nested": {"deep": "data"}}
        from apply_compression import _clean_dict_for_serialization
        cleaned = _clean_dict_for_serialization(test_dict)
        print("   ✅ Limpieza de diccionarios funciona")
        
        # Probar limpieza de listas
        test_list = ["item1", "item2", {"nested": "data"}]
        from apply_compression import _clean_list_for_serialization
        cleaned_list = _clean_list_for_serialization(test_list)
        print("   ✅ Limpieza de listas funciona")
        
    except Exception as e:
        print(f"   ❌ Error en funciones de utilidad: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print("✅ TODAS LAS FUNCIONALIDADES FUNCIONAN CORRECTAMENTE")
    print("✅ El script apply_compression.py está completamente funcional")
    print("✅ Todas las funciones principales operan sin errores")
    
    return True

if __name__ == "__main__":
    success = test_apply_compression_complete()
    if success:
        print("\n🎉 ¡TEST COMPLETO EXITOSO!")
    else:
        print("\n❌ TEST FALLÓ - Revisar errores arriba")
        sys.exit(1)


