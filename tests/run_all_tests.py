#!/usr/bin/env python3
"""
Ejecutor principal de todos los tests
"""
import unittest
import sys
import os
from pathlib import Path
import importlib.util

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def discover_test_files():
    """Descubre todos los archivos de test en el directorio tests/"""
    test_dir = Path(__file__).parent
    test_files = []
    
    # Excluir archivos que requieren dependencias externas o pytest
    excluded_files = {
        'test_compression_engine.py',  # Requiere pytest
        'test_lora_model.py',          # Requiere pytest
        'test_ollama_server.py'        # Requiere servidor Ollama activo
    }
    
    for test_file in test_dir.glob('test_*.py'):
        if test_file.name not in excluded_files:
            test_files.append(test_file)
    
    return test_files

def run_all_tests():
    """Ejecuta todos los tests disponibles"""
    print("🔍 Descubriendo archivos de test...")
    
    # Usar la nueva función de descubrimiento
    test_files = discover_test_files()
    
    if not test_files:
        print("❌ No se encontraron archivos de test")
        return
    
    print(f"📁 Encontrados {len(test_files)} archivos de test:")
    for test_file in test_files:
        print(f"   - {test_file.name}")
    
    print("\n🚀 Ejecutando tests...")
    
    # Ejecutar tests usando unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_file in test_files:
        try:
            # Importar el módulo de test
            module_name = test_file.stem
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Cargar tests del módulo
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            
        except Exception as e:
            print(f"⚠️  Error cargando {test_file.name}: {e}")
            continue
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    print("🚀 Iniciando ejecución de todos los tests...")
    print("=" * 50)
    
    exit_code = run_all_tests()
    
    if exit_code == 0:
        print("\n✅ Todos los tests pasaron exitosamente!")
    else:
        print("\n❌ Algunos tests fallaron. Revisa los errores arriba.")
    
    sys.exit(exit_code)
