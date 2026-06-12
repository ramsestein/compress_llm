#!/usr/bin/env python3
"""Ollama-compatible API server para modelos locales
Allows using models from the models/ folder with the same API as Ollama.
"""

import argparse
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== Data models (compatible with Ollama) ==================


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    keep_alive: Optional[str] = "5m"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    keep_alive: Optional[str] = "5m"


class ModelInfo(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any] = Field(default_factory=dict)


# ================== Ollama-Compatible Server ==================


class OllamaCompactServer:
    """Server compatible with Ollama API."""

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Cache de modelos disponibles
        self.available_models = self._scan_models()

        # Model actualmente cargado
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Servidor iniciado con {len(self.available_models)} modelos disponibles")
        logger.info(f"Device: {self.device}")

    def _scan_models(self) -> Dict[str, Path]:
        """Escanea el directorio de modelos."""
        models = {}

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # Verify que sea un modelo válido
                if (model_dir / "config.json").exists():
                    models[model_dir.name] = model_dir
                    logger.info(f"Modelo encontrado: {model_dir.name}")

        return models

    def list_models(self) -> List[ModelInfo]:
        """Lista modelos disponibles (compatible con ollama list)."""
        models = []

        for name, path in self.available_models.items():
            # Calculate tamaño
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

            # Get fecha de modificación
            mtime = max(f.stat().st_mtime for f in path.rglob("*") if f.is_file())

            # Detalles del modelo
            details = {}
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    details = {
                        "family": config.get("model_type", "unknown"),
                        "parameter_size": (
                            f"{config.get('num_parameters', 0) / 1e9:.1f}B"
                            if "num_parameters" in config
                            else "unknown"
                        ),
                        "quantization_level": "none",  # Por ahora, actualizar cuando tengamos compresión
                    }

            models.append(
                ModelInfo(
                    name=name,
                    modified_at=datetime.fromtimestamp(mtime).isoformat() + "Z",
                    size=size,
                    digest=f"sha256:{hash(name)}",  # Simulado por ahora
                    details=details,
                )
            )

        return models

    def load_model(self, model_name: str, force_device: Optional[str] = None) -> bool:
        """Carga un modelo específico."""
        if model_name not in self.available_models:
            logger.error(f"Modelo {model_name} not found en {self.models_dir}")
            logger.info(f"Modelos disponibles: {list(self.available_models.keys())}")
            return False

        if self.current_model_name == model_name and force_device is None:
            logger.info(f"Modelo {model_name} ya está cargado")
            return True

        try:
            model_path = self.available_models[model_name]
            logger.info(f"Cargando modelo: {model_name}")
            logger.info(f"Ruta: {model_path}")

            # Determinar device
            device = force_device or self.device

            logger.info(f" Device objetivo: {device}")

            # Verify GPU si se solicita
            if device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("GPU solicitada pero CUDA no está disponible. Usando CPU.")
                    device = "cpu"
                else:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(
                        f" GPU disponible: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)"
                    )

            # Liberar modelo anterior si existe
            if self.current_model is not None:
                logger.info(" Liberando modelo anterior...")
                del self.current_model
                del self.current_tokenizer
                torch.cuda.empty_cache()

            # Load tokenizer
            logger.info("Cargando tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )

            # Configure padding token si no existe
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

            # Load modelo
            logger.info(" Cargando modelo...")

            if device == "cuda":
                # Intentar cargar en FP16 para ahorrar memoria
                try:
                    self.current_model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
                    logger.info("Modelo cargado en FP16")
                except Exception as e:
                    logger.warning(f"Error cargando en FP16: {e}")
                    logger.info("Trying cargar en FP32...")
                    self.current_model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.float32,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
            else:
                # CPU siempre en FP32
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                self.current_model = self.current_model.to(device)

            self.current_model_name = model_name
            self.device = device

            # Information del modelo
            total_params = sum(p.numel() for p in self.current_model.parameters())
            logger.info("Modelo cargado exitosamente")
            logger.info(f"Parameters totales: {total_params / 1e9:.2f}B")

            if device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"Memoria GPU: {allocated:.1f}GB / {reserved:.1f}GB reservada")

            return True

        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            return False


# Alias para compatibilidad
OllamaCompatServer = OllamaCompactServer

# ================== Aplicación FastAPI ==================

app = FastAPI(title="Ollama-Compatible Local Model Server")
server = None


@app.on_event("startup")
async def startup_event():
    """Inicializar servidor al arrancar."""
    global server
    logger.info("Ollama-compatible API server iniciado")


@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "Ollama-compatible server running",
        "version": "0.1.0",
        "models": list(server.available_models.keys()) if server else [],
    }


@app.get("/api/tags")
async def list_models():
    """Lista modelos disponibles (compatible con ollama list)."""
    models = server.list_models()
    return {"models": [model.dict() for model in models]}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Genera texto (compatible con ollama generate)."""
    # Verify si el modelo está cargado
    if server.current_model_name != request.model:
        logger.info(f"Modelo solicitado: {request.model}")
        if not server.load_model(request.model):
            logger.error(f"No se pudo cargar el modelo: {request.model}")
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{request.model}' not found o no se pudo cargar. Modelos disponibles: {list(server.available_models.keys())}",
            )

    # Verify que el modelo esté cargado
    if server.current_model is None:
        logger.error("No hay modelo cargado en memoria")
        raise HTTPException(
            status_code=500,
            detail="Error interno: El modelo no está en memoria. Intenta cargar el modelo nuevamente.",
        )

    # Stream o no stream
    if request.stream:
        return StreamingResponse(
            server.generate_stream(request.prompt, request.system, request.options),
            media_type="application/json",
        )
    else:
        # Generación completa (no streaming)
        generated = ""
        async for chunk in server.generate_stream(request.prompt, request.system, request.options):
            data = json.loads(chunk)
            if "response" in data:
                generated += data["response"]

        return {
            "model": request.model,
            "created_at": datetime.now().isoformat() + "Z",
            "response": generated,
            "done": True,
        }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat (compatible con ollama chat)."""
    # Verify modelo
    if server.current_model_name != request.model and not server.load_model(request.model):
        raise HTTPException(status_code=404, detail=f"Modelo '{request.model}' not found")

    if request.stream:
        return StreamingResponse(
            server.chat_stream(request.messages, request.options), media_type="application/json"
        )
    else:
        # Chat completo
        generated = ""
        async for chunk in server.chat_stream(request.messages, request.options):
            data = json.loads(chunk)
            if "message" in data:
                generated += data["message"]["content"]

        return {
            "model": request.model,
            "created_at": datetime.now().isoformat() + "Z",
            "message": {"role": "assistant", "content": generated},
            "done": True,
        }


@app.get("/api/version")
async def version():
    """Versión del servidor."""
    return {"version": "0.1.0"}


# ================== CLI para selección de modelo ==================


def select_model_interactive(available_models: Dict[str, Path]) -> Optional[str]:
    """Permite al usuario seleccionar un modelo interactivamente."""
    if not available_models:
        print("No models found en la carpeta models/")
        return None

    print("\n" + "=" * 60)
    print(" SERVIDOR API COMPATIBLE CON OLLAMA")
    print("=" * 60)
    print("\nModelos disponibles:")
    print("-" * 60)

    models_list = list(available_models.items())
    for i, (name, path) in enumerate(models_list, 1):
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        print(f"  [{i}] {name:<30} ({size:.1f} GB)")

    print("\n  [0] Iniciar sin cargar modelo (cargar bajo demanda)")
    print("-" * 60)

    while True:
        try:
            choice = input("\nSelecciona un modelo (número): ").strip()

            if choice == "0":
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(models_list):
                return models_list[idx][0]
            else:
                print("Selection inválida")
        except (ValueError, KeyboardInterrupt):
            print("\n Saliendo...")
            sys.exit(0)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Ollama-compatible API server para modelos locales"
    )
    parser.add_argument(
        "--port", type=int, default=11435, help="Puerto del servidor (default: 11435)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host del servidor (default: 0.0.0.0)"
    )
    parser.add_argument("--model", type=str, help="Modelo a cargar al inicio")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directorio de modelos (default: ./models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device a usar (default: auto)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Recargar automáticamente en cambios (desarrollo)"
    )
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determinar device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create servidor global
    global server
    server = OllamaCompatServer(args.models_dir)
    server.device = device  # Set device inicial

    # Si no hay modelos disponibles, salir
    if not server.available_models:
        print("\nNo models found en el directorio especificado")
        print(f"Directorio buscado: {Path(args.models_dir).absolute()}")
        print("\nAsegúrate de que:")
        print("   1. El directorio existe")
        print("   2. Contiene modelos de HuggingFace (con config.json)")
        print("   3. La ruta es correcta")
        sys.exit(1)

    # Select modelo inicial
    if args.model:
        if args.model in server.available_models:
            initial_model = args.model
        else:
            print(f"\nModelo '{args.model}' not found")
            print(f"Modelos disponibles: {', '.join(server.available_models.keys())}")
            sys.exit(1)
    else:
        initial_model = select_model_interactive(server.available_models)

    # Load modelo inicial si se seleccionó uno
    if initial_model:
        print(f"\n⏳ Cargando modelo: {initial_model}")
        print(f" Device: {device}")
        if server.load_model(initial_model, force_device=device):
            print(f"Modelo {initial_model} cargado exitosamente")
        else:
            print(f"Error cargando modelo {initial_model}")
            print("\nPosibles soluciones:")
            print("   1. Verificar que el modelo es compatible con transformers")
            print("   2. Verificar memoria disponible (GPU/RAM)")
            print("   3. Probar con --device cpu si hay problemas con GPU")
            print("   4. Usar --verbose para más información")
    else:
        print("\nIniciando servidor sin modelo precargado")
        print("   Los modelos se cargarán bajo demanda según las peticiones")

    # Information del servidor
    print(f"\nServidor iniciado en http://{args.host}:{args.port}")
    print(" Compatible con Ollama API")
    print("\nEjemplos de uso:")
    print(f"   curl http://localhost:{args.port}/api/generate -d '{{")
    print(f'     "model": "{initial_model or list(server.available_models.keys())[0]}",')
    print('     "prompt": "Hola, ¿cómo estás?"')
    print("   }'")
    print("\n   # Con librería Ollama (cambiar puerto):")
    print(f"   ollama = Ollama(host='http://localhost:{args.port}')")
    print("\n   # O usa el script de prueba:")
    print("   python test_ollama_simple.py")
    print("\nPresiona Ctrl+C para detener el servidor")

    # Configure manejador de señales
    def signal_handler(sig, frame):
        print("\n\n Deteniendo servidor...")
        if server.device == "cuda" and server.current_model is not None:
            print(" Liberando memoria GPU...")
            del server.current_model
            torch.cuda.empty_cache()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run servidor
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.verbose else "debug",
    )


if __name__ == "__main__":
    main()
