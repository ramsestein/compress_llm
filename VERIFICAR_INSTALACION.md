# ✅ Verificar Instalación - Compress LLM

## 🔍 Comprobación Rápida

Ejecuta este comando para verificar que todo está instalado correctamente:

```bash
python tests/run_all_tests.py
```

**Resultado esperado:**
```
Tests ejecutados: 261, Errores: 0, Fallos: 0
Tiempo total: ~30 segundos
```

✅ **Si ves "Errores: 0, Fallos: 0" → ¡Todo funciona perfectamente!**

---

## 🔧 Verificaciones Individuales

### 1. Verificar Python
```bash
python --version
# Debe mostrar: Python 3.8.x o superior
```

### 2. Verificar PyTorch
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### 3. Verificar Transformers
```bash
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### 4. Verificar PEFT
```bash
python -c "import peft; print(f'PEFT {peft.__version__}')"
```

---

## 🚨 Si Hay Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Error: "No module named 'transformers'"
```bash
pip install transformers datasets accelerate
```

### Error: "No module named 'peft'"
```bash
pip install peft
```

### Error: "CUDA not available"
- **Normal**: El sistema funcionará en CPU
- **Para GPU**: Instala versión CUDA de PyTorch

---

## 🎯 Test de Funcionalidad

### Test 1: Sistema de Compresión
```bash
python -c "
from create_compress.compression_engine import CompressionEngine
from create_compress.compression_methods import CompressionMethods
print('✅ Sistema de compresión: OK')
"
```

### Test 2: Sistema LoRA
```bash
python -c "
from LoRa_train.peft_methods import LoRALinear
from LoRa_train.peft_methods_config import LoRAConfig, PEFTMethod
print('✅ Sistema LoRA: OK')
"
```

### Test 3: Dataset Manager
```bash
python -c "
from LoRa_train.dataset_manager import DatasetManager
print('✅ Dataset Manager: OK')
"
```

---

## 📊 Información del Sistema

### Verificar Recursos Disponibles:
```bash
python -c "
import psutil
import torch

print(f'RAM Total: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'RAM Disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB' if torch.cuda.is_available() else 'N/A')
"
```

---

## 🎉 Verificación Completa

Si todos los tests pasan, tu sistema está listo para:

- ✅ **Comprimir modelos** grandes
- ✅ **Entrenar con LoRA** y otros métodos
- ✅ **Optimizar memoria** y rendimiento
- ✅ **Crear modelos personalizados**
- ✅ **Servir modelos** via API

---

## 📞 ¿Aún Hay Problemas?

1. **Revisa los logs** de error
2. **Verifica Python** está en PATH
3. **Reinstala dependencias**: `pip install -r requirements.txt --force-reinstall`
4. **Crea nuevo entorno virtual**:
   ```bash
   python -m venv nuevo_venv
   nuevo_venv\Scripts\activate  # Windows
   source nuevo_venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

---

**¡Tu sistema está listo para crear modelos increíbles! 🚀**
