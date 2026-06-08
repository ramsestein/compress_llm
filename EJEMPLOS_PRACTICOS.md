# 🎯 Ejemplos Prácticos - Compress LLM

## 📚 Casos de Uso Comunes

### 🎯 Ejemplo 1: Comprimir un Modelo Grande

**Situación**: Tienes un modelo de 7GB y quieres reducirlo a 2GB

```bash
# 1. Descargar un modelo de ejemplo
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model.save_pretrained('./models/dialogpt_original')
tokenizer.save_pretrained('./models/dialogpt_original')
"

# 2. Comprimir el modelo
python apply_compression.py --model_path "./models/dialogpt_original" --output_path "./models/dialogpt_comprimido"

# 3. Verificar la compresión
python verify_compression.py --model_path "./models/dialogpt_comprimido"
```

**Resultado**: Modelo reducido de ~1.5GB a ~400MB

---

### 🎯 Ejemplo 2: Mejorar un Modelo con LoRA

**Situación**: Quieres que el modelo sea mejor en español

```bash
# 1. Preparar datos en español (archivo CSV)
# Crear archivo: datasets/espanol.csv
# Formato: text,response
# Ejemplo:
# "Hola, ¿cómo estás?", "¡Hola! Estoy muy bien, ¿y tú?"

# 2. Ejecutar fine-tuning
python finetune_peft.py

# 3. Seguir las instrucciones:
# - Seleccionar: LoRA
# - Modelo: microsoft/DialoGPT-small
# - Datos: datasets/espanol.csv
# - Épocas: 3
# - Learning rate: 2e-4
```

**Resultado**: Modelo mejorado para español con solo 10MB adicionales

---

### 🎯 Ejemplo 3: Optimizar para Poca Memoria

**Situación**: Solo tienes 8GB de RAM

```bash
# 1. Usar QLoRA (cuantización + LoRA)
python finetune_peft.py

# 2. Configuración recomendada:
# - Método: QLoRA
# - Bits: 4
# - Rank: 8
# - Batch size: 1
# - Gradient accumulation: 4
```

**Resultado**: Entrenamiento posible en 8GB de RAM

---

### 🎯 Ejemplo 4: Crear un Chatbot Personalizado

**Situación**: Quieres un chatbot para tu empresa

```bash
# 1. Preparar datos de conversaciones
# Crear: datasets/chatbot_empresa.csv
# Formato: 
# "¿Cuál es el horario de atención?", "Nuestro horario es de 9:00 a 18:00"
# "¿Tienen envío gratis?", "Sí, envío gratis en compras superiores a $50"

# 2. Entrenar con IA3 (muy rápido)
python finetune_peft.py
# - Método: IA3
# - Épocas: 1
# - Learning rate: 1e-3

# 3. Probar el modelo
python test_compressed_model.py --model_path "./finetuned_models/mi_chatbot"
```

**Resultado**: Chatbot personalizado en 30 minutos

---

### 🎯 Ejemplo 5: Servidor de Modelos

**Situación**: Quieres servir modelos a través de API

```bash
# 1. Iniciar servidor
python ollama_compact_server.py --models_dir "./models"

# 2. El servidor estará disponible en:
# http://localhost:8000

# 3. Usar con curl:
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"model": "dialogpt_comprimido", "prompt": "Hola, ¿cómo estás?"}'
```

**Resultado**: API REST para tus modelos

---

## 📊 Comparación de Métodos

| Método | Velocidad | Memoria | Calidad | Uso Recomendado |
|--------|-----------|---------|---------|-----------------|
| **LoRA** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General |
| **IA3** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Rápido |
| **BitFit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Muy limitado |
| **QLoRA** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Poca memoria |
| **DoRA** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Máxima calidad |

---

## 🔧 Configuraciones Recomendadas

### Para Principiantes:
```bash
python finetune_peft.py
# - Método: LoRA
# - Rank: 16
# - Alpha: 32
# - Épocas: 3
# - Learning rate: 2e-4
```

### Para Experiencia Media:
```bash
python finetune_peft.py
# - Método: QLoRA
# - Bits: 4
# - Rank: 32
# - Épocas: 5
# - Learning rate: 1e-4
```

### Para Expertos:
```bash
python finetune_peft.py
# - Método: DoRA
# - Rank: 64
# - Alpha: 128
# - Épocas: 10
# - Learning rate: 5e-5
```

---

## 📈 Monitoreo del Progreso

### Durante el Entrenamiento:
- **Loss**: Debe disminuir (ej: 2.5 → 1.2)
- **Accuracy**: Debe aumentar (ej: 0.3 → 0.8)
- **Memory**: No debe exceder tu RAM disponible

### Métricas Esperadas:
- **LoRA**: Loss < 1.5 después de 3 épocas
- **IA3**: Loss < 2.0 después de 1 época
- **QLoRA**: Loss < 1.8 después de 5 épocas

---

## 🚨 Problemas Comunes y Soluciones

### Error: "CUDA out of memory"
```bash
# Solución: Reducir batch size
python finetune_peft.py
# - Batch size: 1
# - Gradient accumulation: 8
```

### Error: "Model too large"
```bash
# Solución: Usar compresión primero
python apply_compression.py --model_path "modelo_grande" --output_path "modelo_pequeño"
```

### Error: "No convergence"
```bash
# Solución: Ajustar learning rate
python finetune_peft.py
# - Learning rate: 1e-5 (más bajo)
# - Épocas: 10 (más tiempo)
```

---

## 🎉 ¡Éxito!

Cuando veas estos mensajes, todo está funcionando:

```
✅ Modelo comprimido exitosamente
✅ Entrenamiento completado
✅ Modelo guardado en: ./finetuned_models/
✅ Pérdida final: 1.234
✅ Precisión final: 0.856
```

**¡Tu modelo está listo para usar! 🚀**
