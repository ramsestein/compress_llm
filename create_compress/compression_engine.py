import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            return module, result
    
    def _calculate_module_size(self, module: nn.Module) -> int:
        """Calcula el tamaño en bytes de un módulo"""
        total_size = 0
        for param in module.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def _no_compression(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """No aplica compresión"""
        return module
    
    def _apply_int8_quantization(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica cuantización INT8"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Simular cuantización INT8
        quantized = QuantizedLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            bits=8
        )
        
        # Cuantizar pesos
        scale, zero_point = self._calculate_quantization_params(module.weight.data, 8)
        quantized.weight_scale.fill_(scale)
        quantized.weight_zero_point.fill_(zero_point)
        quantized.weight_int = self._quantize_tensor(module.weight.data, scale, zero_point, 8)
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()
        
        return quantized
    
    def _apply_int4_quantization(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica cuantización INT4"""
        if not isinstance(module, nn.Linear):
            return module
        
        quantized = QuantizedLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            bits=4
        )
        
        # Proceso similar pero con 4 bits
        scale, zero_point = self._calculate_quantization_params(module.weight.data, 4)
        quantized.weight_scale.fill_(scale)
        quantized.weight_zero_point.fill_(zero_point)
        quantized.weight_int = self._quantize_tensor(module.weight.data, scale, zero_point, 4)
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()
        
        return quantized
    
    def _apply_int2_quantization(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica cuantización binaria/ternaria"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Cuantización ternaria: -1, 0, 1
        ternary = TernaryLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None
        )
        
        # Ternarizar pesos
        threshold = module.weight.data.abs().mean() * strength
        ternary.weight.data = torch.sign(module.weight.data) * (module.weight.data.abs() > threshold)
        ternary.scale = module.weight.data.abs().mean()
        
        if module.bias is not None:
            ternary.bias.data = module.bias.data.clone()
        
        return ternary
    
    def _apply_pruning(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica poda no estructurada"""
        if not isinstance(module, nn.Linear):
            return module
        
        pruned = PrunedLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            sparsity=strength
        )
        
        # Crear máscara de poda
        weights_abs = module.weight.data.abs()
        threshold = torch.quantile(weights_abs.flatten(), strength)
        mask = weights_abs > threshold
        
        pruned.weight.data = module.weight.data * mask
        pruned.mask = mask
        
        if module.bias is not None:
            pruned.bias.data = module.bias.data.clone()
        
        return pruned
    
    def _apply_structured_pruning(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica poda estructurada (canales/neuronas)"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Calcular importancia por neurona de salida
        importance = module.weight.data.abs().sum(dim=1)
        num_keep = int(module.out_features * (1 - strength))
        
        # Seleccionar neuronas más importantes
        _, indices = torch.topk(importance, num_keep)
        indices = indices.sort()[0]
        
        # Crear capa podada
        pruned = nn.Linear(
            module.in_features,
            num_keep,
            bias=module.bias is not None
        )
        
        pruned.weight.data = module.weight.data[indices]
        if module.bias is not None:
            pruned.bias.data = module.bias.data[indices]
        
        # Guardar índices para reconstrucción
        pruned.kept_indices = indices
        
        return pruned
    
    def _apply_tucker_decomposition(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica descomposición Tucker"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Calcular rangos basados en strength
        rank_in = max(1, int(module.in_features * (1 - strength * 0.7)))
        rank_out = max(1, int(module.out_features * (1 - strength * 0.7)))
        
        tucker = TuckerLinear(
            module.in_features,
            module.out_features,
            rank_in,
            rank_out,
            bias=module.bias is not None
        )
        
        # Descomposición SVD para inicialización
        U, S, V = torch.svd(module.weight.data)
        
        tucker.factor_in.data = V[:, :rank_in].T
        tucker.factor_out.data = U[:, :rank_out]
        tucker.core.data = torch.diag(S[:min(rank_in, rank_out)])
        
        if module.bias is not None:
            tucker.bias.data = module.bias.data.clone()
        
        return tucker
    
    def _apply_mpo_decomposition(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica Matrix Product Operators"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Configuración MPO
        bond_dims = config.get('bond_dims', [4, 8, 4])
        bond_dims = [max(1, int(d * (1 - strength))) for d in bond_dims]
        
        mpo = MPOLinear(
            module.in_features,
            module.out_features,
            bond_dims,
            bias=module.bias is not None
        )
        
        # Inicializar con descomposición TT
        mpo.initialize_from_linear(module)
        
        return mpo
    
    def _apply_svd_decomposition(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica descomposición SVD"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Calcular rango basado en strength
        max_rank = min(module.in_features, module.out_features)
        rank = max(1, int(max_rank * (1 - strength)))
        
        svd = SVDLinear(
            module.in_features,
            module.out_features,
            rank,
            bias=module.bias is not None
        )
        
        # Descomposición SVD
        U, S, V = torch.svd(module.weight.data)
        
        svd.U.data = U[:, :rank]
        svd.S.data = S[:rank]
        svd.V.data = V[:, :rank].T
        
        if module.bias is not None:
            svd.bias.data = module.bias.data.clone()
        
        return svd
    
    def _apply_knowledge_distillation(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica destilación de conocimiento"""
        # Esta técnica requiere:
        # 1. Un modelo profesor
        # 2. Datos de calibración
        # 3. Proceso de entrenamiento
        
        # Por ahora, simular con SVD
        logger.info("Knowledge distillation: usando SVD como aproximación")
        return self._apply_svd_decomposition(module, strength * 0.7, config)
    
    def _apply_lora_adaptation(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica adaptadores LoRA"""
        if not isinstance(module, nn.Linear):
            return module
        
        # Calcular rango basado en strength
        max_rank = min(module.in_features, module.out_features) // 4
        rank = max(1, int(max_rank * (1.0 - strength)))
        
        lora = LoRALinear.from_linear(module, rank=rank)
        
        return lora
    
    def _apply_mixed_precision(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica precisión mixta adaptativa"""
        if not isinstance(module, nn.Linear):
            return module
        
        mixed = MixedPrecisionLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None
        )
        
        # Ajustar ratios según strength
        mixed.high_precision_ratio = 0.3 * (1.0 - strength)
        mixed.medium_precision_ratio = 0.4 * (1.0 - strength * 0.5)
        
        mixed.set_weight_importance(module.weight.data)
        
        if module.bias is not None:
            mixed.bias.data = module.bias.data.clone()
        
        return mixed
    
    def _apply_block_sparse(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica sparsidad por bloques"""
        if not isinstance(module, nn.Linear):
            return module
        
        block_size = config.get('block_size', 16)
        
        block_sparse = BlockSparseLinear(
            module.in_features,
            module.out_features,
            block_size=block_size,
            sparsity=strength,
            bias=module.bias is not None
        )
        
        block_sparse.apply_block_sparsity(module.weight.data)
        
        if module.bias is not None:
            block_sparse.bias.data = module.bias.data.clone()
        
        return block_sparse
    
    def _apply_neural_pruning(self, module: nn.Module, strength: float, config: Dict) -> nn.Module:
        """Aplica poda neuronal con red auxiliar"""
        # Similar a pruning pero con decisiones aprendidas
        # Por ahora, usar pruning estándar
        return self._apply_pruning(module, strength, config)
    
    def _calculate_quantization_params(self, tensor: torch.Tensor, bits: int) -> Tuple[float, int]:
        """Calcula parámetros de cuantización"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        qmin = 0
        qmax = 2**bits - 1

        # Evitar división por cero cuando todos los valores son iguales
        if max_val == min_val:
            return 1.0, 0

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        return scale, int(zero_point)
    
    def _quantize_tensor(self, tensor: torch.Tensor, scale: float, zero_point: int, bits: int) -> torch.Tensor:
        """Cuantiza un tensor"""
        qmin = 0
        qmax = 2**bits - 1
        
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized.to(torch.int8 if bits <= 8 else torch.int16)


# Clases auxiliares para diferentes tipos de compresión

class QuantizedLinear(nn.Module):
    """Capa lineal cuantizada"""
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Almacenar pesos cuantizados
        self.register_buffer('weight_int', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Dequantizar pesos
        weight = (self.weight_int - self.weight_zero_point) * self.weight_scale
        return F.linear(x, weight, self.bias)


class TernaryLinear(nn.Module):
    """Capa lineal ternaria"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class PrunedLinear(nn.Linear):
    """Capa lineal con poda"""
    def __init__(self, in_features, out_features, bias=True, sparsity=0.5):
        super().__init__(in_features, out_features, bias)
        self.sparsity = sparsity
        self.register_buffer('mask', torch.ones_like(self.weight))
    
    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class TuckerLinear(nn.Module):
    """Capa lineal con descomposición Tucker"""
    def __init__(self, in_features, out_features, rank_in, rank_out, bias=True):
        super().__init__()
        self.factor_in = nn.Parameter(torch.randn(rank_in, in_features))
        self.core = nn.Parameter(torch.randn(rank_out, rank_in))
        self.factor_out = nn.Parameter(torch.randn(out_features, rank_out))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        x = F.linear(x, self.factor_in)
        x = F.linear(x, self.core)
        x = F.linear(x, self.factor_out.T, self.bias)
        return x


class SVDLinear(nn.Module):
    """Capa lineal con descomposición SVD"""
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.S = nn.Parameter(torch.randn(rank))
        self.V = nn.Parameter(torch.randn(rank, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        weight = self.U @ torch.diag(self.S) @ self.V
        return F.linear(x, weight, self.bias)


class LoRALinear(nn.Module):
    """Capa lineal con adaptadores LoRA"""
    def __init__(self, in_features, out_features, rank=16, bias=True):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.scaling = 1.0 / rank
        
        # Inicialización
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    @staticmethod
    def from_linear(linear_layer, rank=16):
        """Crea LoRALinear desde una capa Linear existente"""
        lora = LoRALinear(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            bias=linear_layer.bias is not None
        )
        lora.base_layer.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            lora.base_layer.bias.data = linear_layer.bias.data.clone()
        return lora
    
    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B.T) * self.scaling
        return base_out + lora_out


class MixedPrecisionLinear(nn.Module):
    """Capa lineal con precisión mixta"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Ratios de precisión
        self.high_precision_ratio = 0.2
        self.medium_precision_ratio = 0.3
        # El resto es baja precisión
        
        # Máscaras para diferentes precisiones
        self.register_buffer('high_precision_mask', torch.zeros(out_features, in_features, dtype=torch.bool))
        self.register_buffer('medium_precision_mask', torch.zeros(out_features, in_features, dtype=torch.bool))
        
        # Pesos en diferentes precisiones
        self.weight_high = nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer('weight_medium', torch.zeros(out_features, in_features, dtype=torch.float16))
        self.register_buffer('weight_low', torch.zeros(out_features, in_features, dtype=torch.int8))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def set_weight_importance(self, weight):
        """Establece máscaras basadas en importancia de pesos"""
        importance = weight.abs()
        flat_importance = importance.flatten()
        
        # Umbrales para diferentes precisiones
        high_threshold = torch.quantile(flat_importance, 1 - self.high_precision_ratio)
        medium_threshold = torch.quantile(flat_importance, 1 - self.high_precision_ratio - self.medium_precision_ratio)
        
        self.high_precision_mask = importance > high_threshold
        self.medium_precision_mask = (importance > medium_threshold) & ~self.high_precision_mask
        
        # Asignar pesos
        self.weight_high.data = weight * self.high_precision_mask
        self.weight_medium = (weight * self.medium_precision_mask).half()
        
        # Cuantizar resto a INT8
        low_mask = ~self.high_precision_mask & ~self.medium_precision_mask
        low_weights = weight * low_mask
        scale = low_weights.abs().max() / 127.0
        self.weight_low = (low_weights / scale).round().to(torch.int8)
        self.register_buffer('low_scale', torch.tensor(scale))
    
    def forward(self, x):
        # Combinar pesos de diferentes precisiones
        weight = self.weight_high
        weight = weight + self.medium_precision_mask * self.weight_medium.float()
        
        low_mask = ~self.high_precision_mask & ~self.medium_precision_mask
        weight = weight + low_mask * (self.weight_low.float() * self.low_scale)
        
        return F.linear(x, weight, self.bias)


class BlockSparseLinear(nn.Module):
    """Capa lineal con sparsidad por bloques"""
    def __init__(self, in_features, out_features, block_size=16, sparsity=0.5, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.sparsity = sparsity
        
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer('block_mask', torch.ones(
            (out_features + block_size - 1) // block_size,
            (in_features + block_size - 1) // block_size,
            dtype=torch.bool
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def apply_block_sparsity(self, weight):
        """Aplica sparsidad por bloques basada en importancia"""
        self.weight.data = weight.clone()
        
        # Calcular importancia por bloque
        block_importance = []
        for i in range(0, self.out_features, self.block_size):
            for j in range(0, self.in_features, self.block_size):
                block = weight[i:i+self.block_size, j:j+self.block_size]
                importance = block.abs().sum().item()
                block_importance.append((importance, i//self.block_size, j//self.block_size))
        
        # Ordenar bloques por importancia
        block_importance.sort(reverse=True)
        
        # Mantener solo los bloques más importantes
        num_keep = int(len(block_importance) * (1 - self.sparsity))
        self.block_mask.fill_(False)
        
        for _, bi, bj in block_importance[:num_keep]:
            self.block_mask[bi, bj] = True
        
        # Aplicar máscara
        self._apply_mask()
    
    def _apply_mask(self):
        """Aplica la máscara de bloques a los pesos"""
        for i in range(self.block_mask.shape[0]):
            for j in range(self.block_mask.shape[1]):
                if not self.block_mask[i, j]:
                    start_i = i * self.block_size
                    start_j = j * self.block_size
                    end_i = min(start_i + self.block_size, self.out_features)
                    end_j = min(start_j + self.block_size, self.in_features)
                    self.weight.data[start_i:end_i, start_j:end_j] = 0
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class MPOLinear(nn.Module):
    """Capa lineal con Matrix Product Operators"""
    def __init__(self, in_features, out_features, bond_dims, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dims = bond_dims
        
        # Crear tensores MPO
        self.cores = nn.ParameterList()
        
        # Dimensiones para cada core
        dims = [1] + bond_dims + [1]
        n_cores = len(bond_dims) + 1
        
        # Distribuir features entre cores
        in_dims = self._distribute_features(in_features, n_cores)
        out_dims = self._distribute_features(out_features, n_cores)
        
        for i in range(n_cores):
            core = nn.Parameter(torch.randn(
                dims[i], out_dims[i], in_dims[i], dims[i+1]
            ))
            self.cores.append(core)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def _distribute_features(self, total, n_parts):
        """Distribuye features entre las partes"""
        base = total // n_parts
        remainder = total % n_parts
        
        dims = [base] * n_parts
        for i in range(remainder):
            dims[i] += 1
        
        return dims
    
    def initialize_from_linear(self, linear_layer):
        """Inicializa desde una capa lineal usando descomposición TT"""
        # Implementación simplificada
        # En práctica, usar tt-decomposition proper
        weight = linear_layer.weight.data
        
        # Por ahora, inicialización aleatoria
        for core in self.cores:
            nn.init.xavier_uniform_(core)
        
        if linear_layer.bias is not None and self.bias is not None:
            self.bias.data = linear_layer.bias.data.clone()
    
    def forward(self, x):
        # Implementación simplificada de MPO forward
        # En práctica, esto requiere reshape y contracciones tensoriales
        batch_size = x.shape[0]
        
        # Por ahora, usar aproximación con producto de matrices
        weight = self._reconstruct_weight()
        return F.linear(x, weight, self.bias)
    
    def _reconstruct_weight(self):
        """Reconstruye la matriz de pesos desde cores MPO"""
        # Implementación simplificada
        # En práctica, esto requiere contracciones tensoriales apropiadas
        
        # Por ahora, retornar matriz aleatoria del tamaño correcto
        return torch.randn(self.out_features, self.in_features, device=self.cores[0].device)
