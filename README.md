# 🚀 Compress LLM - Sistema de Compresión y Fine-tuning de Modelos de Lenguaje

Un sistema completo para comprimir y mejorar modelos de lenguaje usando técnicas avanzadas de compresión y fine-tuning.

## 📋 ¿Qué hace este proyecto?

Este proyecto te permite:

- **Comprimir modelos grandes** para que ocupen menos espacio y sean más rápidos
- **Mejorar modelos** con técnicas de fine-tuning como LoRA, IA3, BitFit, etc.
- **Optimizar el uso de memoria** en tu computadora
- **Crear modelos personalizados** para tareas específicas

## 🛠️ Requisitos del Sistema

### Mínimos:
- **Windows 10/11** o **Linux** o **macOS**
- **Python 3.8** o superior
- **8 GB de RAM** (16 GB recomendado)
- **10 GB de espacio libre** en disco

### Recomendados:
- **GPU NVIDIA** con 8+ GB de VRAM (para entrenamiento más rápido)
- **32 GB de RAM** (para modelos grandes)
- **100 GB de espacio libre** (para múltiples modelos)

## 📦 Instalación Paso a Paso

### 1. Descargar el Proyecto

```bash
# Opción 1: Clonar desde Git
git clone https://github.com/tu-usuario/compress_llm_git.git
cd compress_llm_git

# Opción 2: Descargar ZIP
# Descarga el archivo ZIP desde GitHub y extráelo
```

### 2. Instalar Dependencias

Abre una terminal (Command Prompt en Windows) en la carpeta del proyecto y ejecuta:

```bash
# Instalar todas las dependencias automáticamente
python install.py
```

**O manualmente:**

```bash
# Crear entorno virtual
python -m venv comp_venv

# Activar entorno virtual
# En Windows:
comp_venv\Scripts\activate
# En Linux/Mac:
source comp_venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 🎯 Cómo Usar el Sistema

### Opción 1: Interfaz Interactiva (Recomendado para principiantes)

```bash
# Ejecutar el asistente interactivo
python finetune_peft.py
```

Esto te guiará paso a paso para:
- Seleccionar tu modelo
- Elegir el método de fine-tuning
- Configurar parámetros
- Iniciar el entrenamiento

### Opción 2: Scripts Específicos

#### Para Comprimir un Modelo:

```bash
# Comprimir un modelo existente
python apply_compression.py --model_path "ruta/al/modelo" --output_path "modelo_comprimido"
```

#### Para Fine-tuning con LoRA:

```bash
# Entrenar con LoRA
python finetune_lora.py
```

#### Para Verificar un Modelo:

```bash
# Verificar que un modelo funciona correctamente
python test_compressed_model.py --model_path "ruta/al/modelo"
```

## 🔧 Métodos Disponibles

### Métodos de Compresión:
- **INT8/INT4/INT2**: Reduce la precisión de los números
- **Pruning**: Elimina conexiones innecesarias
- **SVD/Tucker/MPO**: Descompone matrices grandes
- **Mixed Precision**: Usa diferentes precisiones según necesidad

### Métodos de Fine-tuning:
- **LoRA**: Mejora eficiente con matrices de bajo rango
- **IA3**: Adaptación rápida con pocos parámetros
- **BitFit**: Solo entrena los sesgos
- **Adapter**: Inserta capas adaptativas
- **QLoRA**: LoRA con cuantización
- **DoRA**: LoRA con descomposición de magnitud
- **MoLoRA**: LoRA con múltiples expertos
- **Compacter/KronA/S4/Houlsby**: Métodos avanzados de adaptación

## 📁 Estructura del Proyecto

```
compress_llm_git/
├── 📁 create_compress/          # Sistema de compresión
├── 📁 LoRa_train/              # Sistema de fine-tuning
├── 📁 datasets/                # Datos de entrenamiento
├── 📁 models/                  # Modelos guardados
├── 📁 finetuned_models/        # Modelos entrenados
├── 📁 compression_analysis/    # Reportes de compresión
├── 📁 tests/                   # Tests del sistema
├── 🐍 finetune_peft.py        # Interfaz principal
├── 🐍 apply_compression.py    # Script de compresión
├── 🐍 finetune_lora.py        # Script de LoRA
└── 📄 requirements.txt         # Dependencias
```

## 🚀 Ejemplos de Uso

### Ejemplo 1: Comprimir un Modelo

```bash
# 1. Descargar un modelo (ejemplo con DialoGPT)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small').save_pretrained('./models/mi_modelo')"

# 2. Comprimir el modelo
python apply_compression.py --model_path "./models/mi_modelo" --output_path "./models/mi_modelo_comprimido"
```

### Ejemplo 2: Fine-tuning con LoRA

```bash
# 1. Preparar datos de entrenamiento
# Coloca tus archivos .csv en la carpeta datasets/

# 2. Ejecutar fine-tuning
python finetune_peft.py
# Sigue las instrucciones en pantalla
```

### Ejemplo 3: Servidor Ollama

```bash
# Crear un servidor compatible con Ollama
python ollama_compact_server.py --models_dir "./models"
```

## ⚠️ Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Error: "CUDA out of memory"
- Reduce el tamaño del batch
- Usa un modelo más pequeño
- Cierra otras aplicaciones que usen GPU

### Error: "Not enough disk space"
- Libera espacio en disco
- Usa compresión más agresiva
- Elimina modelos temporales

### Error: "Python not found"
- Asegúrate de que Python esté en el PATH
- Reinstala Python marcando "Add to PATH"

## 📊 Monitoreo y Reportes

El sistema genera automáticamente:

- **Reportes de compresión**: Análisis detallado de la compresión
- **Métricas de entrenamiento**: Pérdida, precisión, etc.
- **Uso de memoria**: Estadísticas de RAM y VRAM
- **Tiempos de ejecución**: Duración de cada proceso

Los reportes se guardan en:
- `compression_analysis/` para análisis de compresión
- `finetuned_models/` para resultados de entrenamiento

## 🔒 Seguridad y Privacidad

- **Datos locales**: Todo se procesa en tu computadora
- **Sin conexión**: No se envían datos a servidores externos
- **Modelos propios**: Puedes usar tus propios modelos y datos

## 🤝 Contribuir

Para contribuir al proyecto:

1. Haz un fork del repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Abre un Pull Request

## 📞 Soporte

Si tienes problemas:

1. **Revisa esta documentación**
2. **Ejecuta los tests**: `python tests/run_all_tests.py`
3. **Busca en Issues** de GitHub
4. **Crea un nuevo Issue** con detalles del problema

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Hugging Face por las librerías de transformers
- Microsoft por DialoGPT
- La comunidad de open source

---

**¡Disfruta comprimiendo y mejorando tus modelos de lenguaje! 🎉**
