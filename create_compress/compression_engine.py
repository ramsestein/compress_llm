import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from .compression_methods import apply_compression as _apply_compression


@dataclass
class CompressionResult:
    """Resultado básico de aplicar compresión"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    method_used: str
    success: bool
    error: Optional[str] = None


class CompressionEngine:
    """Motor ligero para aplicar técnicas de compresión a capas individuales.

    Este motor actúa como un contenedor fino sobre las funciones de
    ``compression_methods``.  Su objetivo es proporcionar una interfaz estable
    para ``apply_compression.py`` sin incluir lógica compleja ni dependencias
    innecesarias.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------
    def apply_method(
        self,
        module: nn.Module,
        method_name: str,
        strength: float,
        layer_config: Dict[str, Any],
    ) -> nn.Module:
        """Aplica un único método de compresión a ``module``.

        Parameters
        ----------
        module:
            Capa del modelo a modificar.
        method_name:
            Nombre del método (por ejemplo ``int8_quantization``).
        strength:
            Intensidad del método entre 0 y 1.
        layer_config:
            Configuración adicional; sólo se utiliza la clave ``params`` si
            está presente.
        """
        method_config = {"name": method_name, "strength": strength}
        if "params" in layer_config and isinstance(layer_config["params"], dict):
            method_config.update(layer_config["params"])
        return _apply_compression(module, method_config, self.device)

    def compress_layer(
        self, module: nn.Module, layer_config: Dict[str, Any]
    ) -> Tuple[nn.Module, CompressionResult]:
        """Aplica secuencialmente los métodos definidos en ``layer_config``.

        Returns el módulo posiblemente modificado junto con estadísticas
        simples de compresión.
        """
        original_size = self._module_size(module)
        methods = layer_config.get("methods", [])

        try:
            for method in methods:
                module = self.apply_method(
                    module,
                    method.get("name", "none"),
                    method.get("strength", 0.0),
                    layer_config,
                )
            compressed_size = self._module_size(module)
            ratio = 1 - (compressed_size / original_size) if original_size else 0.0
            return module, CompressionResult(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=ratio,
                method_used=",".join(m.get("name", "none") for m in methods),
                success=True,
            )
        except Exception as exc:  # pragma: no cover - logging
            return module, CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=0.0,
                method_used=",".join(m.get("name", "none") for m in methods),
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _module_size(self, module: nn.Module) -> int:
        """Calcula el tamaño en bytes de ``module``."""
        return sum(p.numel() * p.element_size() for p in module.parameters())
