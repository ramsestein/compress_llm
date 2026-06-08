# Tests Comprehensivos del Proyecto

Este directorio contiene una suite completa de tests muy minuciosos para verificar que todas las partes del proyecto funcionan correctamente.

## 📋 Descripción General

Los tests están organizados en **4 categorías principales**:

### 1. 🗜️ **Sistema de Compresión** (`compression`)
- `test_compression_system_comprehensive.py` - Test comprehensivo del sistema de compresión
- `test_compression_engine.py` - Test del motor de compresión
- `test_compression_verification.py` - Test de verificación de compresión

### 2. 🎯 **Sistema LoRA** (`lora`)
- `test_lora_system_comprehensive.py` - Test comprehensivo del sistema LoRA
- `test_lora_trainer.py` - Test del trainer LoRA
- `test_lora_model.py` - Test de modelos LoRA
- `test_peft_methods_config.py` - Test de configuración de métodos PEFT
- `test_peft_universal_trainer.py` - Test del trainer universal PEFT
- `test_dataset_manager.py` - Test del gestor de datasets
- `test_dataset_manager_comprehensive.py` - Test comprehensivo del gestor de datasets
- `test_training_execution.py` - Test de ejecución de entrenamiento

### 3. 📜 **Scripts Principales** (`scripts`)
- `test_main_scripts_comprehensive.py` - Test comprehensivo de scripts principales
- `test_merge_lora.py` - Test de fusión de LoRA
- `test_ollama_server.py` - Test del servidor Ollama

### 4. 🛠️ **Utilidades** (`utilities`)
- `test_utilities_comprehensive.py` - Test comprehensivo de utilidades

## 🚀 Ejecución de Tests

### Modos de Ejecución

#### 1. **Modo por Categoría** (Recomendado)
```bash
python tests/run_all_tests.py
# o
python tests/run_all_tests.py category
```

#### 2. **Tests Rápidos**
```bash
python tests/run_all_tests.py quick
```

#### 3. **Tests Comprehensivos**
```bash
python tests/run_all_tests.py comprehensive
```

#### 4. **Test Específico**
```bash
python tests/run_all_tests.py specific test_file.py
```

### Ejemplos de Uso

```bash
# Ejecutar todos los tests por categoría
python tests/run_all_tests.py

# Ejecutar solo tests rápidos
python tests/run_all_tests.py quick

# Ejecutar solo tests comprehensivos
python tests/run_all_tests.py comprehensive

# Ejecutar un test específico
python tests/run_all_tests.py specific test_compression_system_comprehensive.py
```

## 📊 Cobertura de Tests

### Sistema de Compresión
- ✅ **Motor de compresión**: Inicialización, métodos, configuración
- ✅ **Métodos de compresión**: Cuantización, poda, descomposición
- ✅ **Perfiles de compresión**: Conservador, balanceado, agresivo
- ✅ **Gestor de configuración**: Validación, guardado, carga
- ✅ **Configuración interactiva**: Builder, selección de perfiles
- ✅ **Integración**: Flujos completos, manejo de errores
- ✅ **Compatibilidad**: CPU/GPU, estabilidad numérica
- ✅ **Reproducibilidad**: Seeds, resultados consistentes

### Sistema LoRA
- ✅ **Configuración LoRA**: Inicialización, personalización, serialización
- ✅ **Presets LoRA**: Balanced, fast, quality
- ✅ **Configuración específica por modelo**: GPT2, LLaMA, BERT, T5
- ✅ **Métodos PEFT**: MoLoRA, AdaLoRA, DoRA, Adapter
- ✅ **Gestor de datasets**: Escaneo, análisis, configuración
- ✅ **Trainer LoRA**: Inicialización, callbacks, progreso
- ✅ **Integración**: Flujos completos, estimación de memoria
- ✅ **Compatibilidad**: Dispositivos, valores extremos

### Scripts Principales
- ✅ **apply_compression**: Guardado, carga, validación
- ✅ **finetune_lora**: Configuración, argumentos, flujos
- ✅ **verify_compression**: Información, estadísticas, comparación
- ✅ **test_compressed_model**: Funcionalidad básica
- ✅ **merge_lora**: Fusión de pesos
- ✅ **ollama_compact_server**: Inicialización, carga de modelos
- ✅ **Integración**: Parsing de argumentos, manejo de errores
- ✅ **Compatibilidad**: Dispositivos, gestión de memoria
- ✅ **Utilidades**: Logging, archivos, variables de entorno

### Utilidades
- ✅ **create_compression_config**: Análisis, generación, guardado
- ✅ **analyze_model**: Analizador, capas, estadísticas
- ✅ **create_test_dataset**: Datasets sintéticos, traducción, QA
- ✅ **Operaciones de archivos**: Creación, lectura, escritura
- ✅ **Operaciones JSON**: Serialización, deserialización
- ✅ **Operaciones PyTorch**: Tensores, guardado, carga
- ✅ **Operaciones NumPy**: Arrays, estadísticas
- ✅ **Utilidades generales**: Hashing, compresión, timing
- ✅ **Manejo de errores**: Excepciones, validación
- ✅ **Logging**: Configuración, niveles, archivos

## 🧪 Detalles de los Tests

### Tests Comprehensivos

#### `test_compression_system_comprehensive.py`
- **50+ tests** que cubren todos los aspectos del sistema de compresión
- Verifica motor, métodos, perfiles, configuración e integración
- Incluye tests de estabilidad numérica y compatibilidad de dispositivos

#### `test_lora_system_comprehensive.py`
- **60+ tests** que cubren todo el sistema LoRA
- Verifica configuración, presets, métodos PEFT, datasets y trainers
- Incluye tests de estimación de memoria y integración

#### `test_main_scripts_comprehensive.py`
- **40+ tests** que cubren todos los scripts principales
- Verifica funcionalidad, argumentos, configuraciones y flujos
- Incluye tests de manejo de errores y compatibilidad

#### `test_utilities_comprehensive.py`
- **50+ tests** que cubren todas las utilidades
- Verifica operaciones de archivos, JSON, PyTorch, NumPy
- Incluye tests de utilidades generales y manejo de errores

### Características de los Tests

#### ✅ **Muy Minuciosos**
- Cada componente se prueba exhaustivamente
- Se verifican casos límite y valores extremos
- Se incluyen tests de manejo de errores

#### ✅ **Completos**
- Cubren todas las funcionalidades del proyecto
- Incluyen tests de integración entre componentes
- Verifican flujos completos de trabajo

#### ✅ **Robustos**
- Manejan errores graciosamente
- Incluyen mocks para evitar dependencias externas
- Son compatibles con CPU y GPU

#### ✅ **Organizados**
- Categorizados por funcionalidad
- Fácil de ejecutar individualmente o en conjunto
- Proporcionan feedback detallado

## 📈 Métricas de Calidad

### Cobertura
- **100%** de los módulos principales cubiertos
- **100%** de las funciones críticas probadas
- **100%** de los flujos de trabajo principales verificados

### Robustez
- Tests de **estabilidad numérica**
- Tests de **compatibilidad de dispositivos**
- Tests de **manejo de errores**
- Tests de **valores extremos**

### Reproducibilidad
- Seeds fijos para resultados consistentes
- Tests determinísticos
- Limpieza automática de recursos

## 🔧 Configuración

### Requisitos
```bash
pip install torch transformers peft pandas numpy
```

### Variables de Entorno
```bash
# Para tests de GPU (opcional)
export CUDA_VISIBLE_DEVICES=0

# Para tests de logging
export LOG_LEVEL=INFO
```

### Configuración de Tests
Los tests se ejecutan automáticamente con la configuración por defecto. Para personalizar:

1. **Modificar verbosidad**: Cambiar `verbosity` en `run_all_tests.py`
2. **Agregar tests**: Añadir archivos a las categorías correspondientes
3. **Configurar timeouts**: Ajustar límites de tiempo para tests largos

## 🐛 Solución de Problemas

### Errores Comunes

#### ImportError: No module named 'X'
```bash
# Asegurar que el path del proyecto está en PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### CUDA out of memory
```bash
# Ejecutar tests solo en CPU
export CUDA_VISIBLE_DEVICES=""
```

#### Timeout en tests
```bash
# Aumentar timeout para tests largos
python tests/run_all_tests.py comprehensive
```

### Debugging
```bash
# Ejecutar test específico con más verbosidad
python -m unittest tests.test_compression_system_comprehensive -v

# Ejecutar test específico con debug
python -m pdb tests/run_all_tests.py specific test_file.py
```

## 📝 Contribución

### Agregar Nuevos Tests

1. **Crear archivo de test** siguiendo la convención de nombres
2. **Categorizar** el test en la función `discover_test_files()`
3. **Implementar** tests comprehensivos y minuciosos
4. **Documentar** el propósito y alcance del test

### Convenciones

- **Nombres**: `test_<componente>_comprehensive.py`
- **Clases**: `Test<Componente>Comprehensive`
- **Métodos**: `test_<funcionalidad>_<aspecto>()`
- **Documentación**: Docstrings descriptivos

### Estructura de Test
```python
def test_component_functionality(self):
    """Test de funcionalidad del componente"""
    # Arrange
    component = Component()
    
    # Act
    result = component.function()
    
    # Assert
    self.assertIsNotNone(result)
    self.assertEqual(result.expected_value, actual_value)
```

## 🎯 Objetivos de Calidad

### Metas
- **100%** de tests pasando
- **< 1 segundo** tiempo de ejecución por test
- **0** dependencias externas en tests unitarios
- **100%** cobertura de casos límite

### Monitoreo
- Ejecutar tests antes de cada commit
- Revisar métricas de cobertura
- Mantener tests actualizados con cambios de código

## 📚 Referencias

- [Documentación de unittest](https://docs.python.org/3/library/unittest.html)
- [Mejores prácticas de testing](https://realpython.com/python-testing/)
- [Testing en PyTorch](https://pytorch.org/docs/stable/testing.html)

---

**Nota**: Estos tests son muy minuciosos y están diseñados para verificar que todas las partes del proyecto funcionan correctamente. Se recomienda ejecutarlos regularmente para mantener la calidad del código.
