"""
Métodos de compresión optimizados para modelos de lenguaje
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union, List
from functools import lru_cache
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Intentar importar TensorLy para MPO
try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train, tucker, Tucker
    tl.set_backend('pytorch')
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False
    logger.warning("TensorLy no está instalado. MPO no estará disponible.")
    logger.warning("Instala con: pip install tensorly")
except Exception as e:
    TENSORLY_AVAILABLE = False
    logger.warning(f"Error al configurar TensorLy: {e}")

# Cache global para métodos compilados
_COMPILED_METHODS_CACHE = {}

class CompressionMethod(ABC):
    """Clase base para métodos de compresión"""
    
    @abstractmethod
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica compresión al módulo"""
        pass
    
    @abstractmethod
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima el ratio de compresión sin aplicar"""
        pass

class QuantizationMethod(CompressionMethod):
    """Cuantización optimizada INT8/INT4/INT2"""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.scale_cache = {}
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica cuantización al módulo"""
        if not hasattr(module, 'weight'):
            return module
        
        # Obtener strength del config
        strength = config.get('strength', 0.5)
        
        # Para PyTorch 2.0+, usar cuantización nativa si es posible
        if hasattr(torch, 'compile') and device.type == 'cuda' and self.bits >= 4:
            return self._compress_pytorch2(module, strength, device)
        else:
            return self._compress_legacy(module, strength, device)
    
    def _compress_pytorch2(self, module: nn.Module, strength: float, 
                          device: torch.device) -> nn.Module:
        """Cuantización optimizada para PyTorch 2.0+"""
        try:
            # Usar dynamic quantization para mejor rendimiento
            if isinstance(module, nn.Linear):
                # Configurar backend
                torch.backends.quantized.engine = 'qnnpack'
                
                # Aplicar cuantización dinámica
                quantized = torch.quantization.quantize_dynamic(
                    module, 
                    {nn.Linear}, 
                    dtype=torch.qint8 if self.bits == 8 else torch.quint4x2
                )
                return quantized
        except Exception as e:
            logger.debug(f"Fallback a cuantización legacy: {e}")
            return self._compress_legacy(module, strength, device)
    
    def _compress_legacy(self, module: nn.Module, strength: float, 
                        device: torch.device) -> nn.Module:
        """Cuantización manual para compatibilidad"""
        with torch.no_grad():
            weight = module.weight.data
            
            # Convertir a float32 si es necesario
            original_dtype = weight.dtype
            if weight.dtype not in [torch.float32, torch.float64]:
                weight = weight.float()
            
            # Calcular escala y zero point según bits
            if self.bits == 2:
                # Cuantización ternaria {-1, 0, 1}
                threshold = weight.abs().mean()
                weight_q = torch.sign(weight) * (weight.abs() > threshold).float()
                weight_dq = weight_q * threshold
            elif self.bits == 4:
                qmin, qmax = -8, 7
                scale = (weight.max() - weight.min()) / (qmax - qmin)
                zero_point = qmin - weight.min() / scale
                weight_q = torch.clamp(torch.round(weight / scale + zero_point), qmin, qmax)
                weight_dq = (weight_q - zero_point) * scale
            else:  # 8 bits
                qmin, qmax = -128, 127
                scale = (weight.max() - weight.min()) / (qmax - qmin)
                zero_point = qmin - weight.min() / scale
                weight_q = torch.clamp(torch.round(weight / scale + zero_point), qmin, qmax)
                weight_dq = (weight_q - zero_point) * scale
            
            # Aplicar strength (mezcla con original)
            weight_final = weight * (1 - strength) + weight_dq * strength
            
            # Restaurar dtype original
            module.weight.data = weight_final.to(original_dtype)
        
        return module
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión por cuantización"""
        if not hasattr(module, 'weight'):
            return 0.0
        
        original_bits = 16  # FP16
        compressed_bits = self.bits
        return 1 - (compressed_bits / original_bits)

class PruningMethod(CompressionMethod):
    """Poda optimizada de pesos"""
    
    def __init__(self, structured: bool = False):
        self.structured = structured
        self._importance_cache = {}
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica poda al módulo"""
        if not hasattr(module, 'weight'):
            return module
        
        strength = config.get('strength', 0.5)
        
        if self.structured:
            return self._structured_pruning(module, strength)
        else:
            return self._magnitude_pruning(module, strength)
    
    def _magnitude_pruning(self, module: nn.Module, strength: float) -> nn.Module:
        """Poda por magnitud optimizada con manejo correcto de dtype"""
        with torch.no_grad():
            weight = module.weight.data
            
            # Convertir a float32 si es necesario para quantile
            original_dtype = weight.dtype
            if weight.dtype not in [torch.float32, torch.float64]:
                weight_for_quantile = weight.float()
            else:
                weight_for_quantile = weight
            
            # Calcular umbral de poda de forma eficiente
            weight_flat = weight_for_quantile.abs().flatten()
            
            # Para tensores grandes, usar muestreo
            if weight_flat.numel() > 1e6:
                # Muestrear 100k elementos
                indices = torch.randint(0, weight_flat.numel(), (100000,))
                sample = weight_flat[indices]
                threshold = torch.quantile(sample, strength)
            else:
                threshold = torch.quantile(weight_flat, strength)
            
            # Crear máscara de poda (usando el dtype original)
            mask = weight.abs() > threshold.to(original_dtype)
            
            # Aplicar máscara
            module.weight.data *= mask.to(original_dtype)
            
            # Guardar máscara para inferencia eficiente
            module.register_buffer('pruning_mask', mask)
        
        return module
    
    def _structured_pruning(self, module: nn.Module, strength: float) -> nn.Module:
        """Poda estructurada (canales/filtros completos)"""
        if not isinstance(module, nn.Linear):
            return self._magnitude_pruning(module, strength)
        
        with torch.no_grad():
            weight = module.weight.data
            
            # Calcular importancia por fila (neurona de salida)
            importance = weight.abs().mean(dim=1)
            
            # Seleccionar filas a mantener
            k = int((1 - strength) * weight.shape[0])
            if k == 0:
                k = 1  # Mantener al menos una neurona
            
            _, indices = torch.topk(importance, k)
            
            # Crear nuevo módulo con menos neuronas
            new_module = nn.Linear(
                module.in_features,
                k,
                bias=module.bias is not None
            )
            
            # Copiar pesos seleccionados
            new_module.weight.data = weight[indices]
            if module.bias is not None:
                new_module.bias.data = module.bias.data[indices]
            
            return new_module
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión por poda"""
        strength = config.get('strength', 0.5)
        return strength  # Directamente el porcentaje podado

class LowRankApproximation(CompressionMethod):
    """Aproximación de bajo rango optimizada - NO crea módulos custom"""
    
    def __init__(self):
        self._svd_cache = {}
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica descomposición de bajo rango IN-PLACE"""
        if not isinstance(module, nn.Linear):
            return module
        
        strength = config.get('strength', 0.5)
        rank_ratio = 1 - strength
        
        with torch.no_grad():
            weight = module.weight.data
            original_dtype = weight.dtype
            
            # Calcular rango objetivo
            min_dim = min(weight.shape)
            target_rank = max(1, int(min_dim * rank_ratio))
            
            try:
                # Convertir a float32 si es necesario para SVD
                if weight.dtype not in [torch.float32, torch.float64]:
                    weight = weight.float()
                
                # SVD truncado eficiente
                if weight.numel() > 1e6:
                    U, S, V = self._randomized_svd(weight, target_rank)
                else:
                    U, S, V = torch.svd_lowrank(weight, q=target_rank)
                
                # Reconstruir peso aproximado evitando la creación explícita de
                # una matriz diagonal grande.  ``torch.diag`` puede ser costoso
                # y en algunos casos provoca problemas de forma cuando las
                # orientaciones de las matrices no coinciden exactamente.  Al
                # multiplicar directamente por ``S`` aprovechamos el broadcast
                # para escalar las columnas de ``U`` antes de aplicar ``V``.
                weight_approx = (U * S) @ V.t()
                
                # Aplicar directamente al módulo original
                module.weight.data = weight_approx.to(original_dtype)
                
                logger.debug(f"SVD aplicado: {weight.shape} -> rango {target_rank}")
                
            except Exception as e:
                logger.warning(f"Error en SVD, manteniendo peso original: {e}")
                # No hacer nada, mantener el peso original
        
        return module  # Devolver el mismo módulo modificado
    
    def _randomized_svd(self, matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, ...]:
        """SVD aleatorizado para matrices grandes"""
        m, n = matrix.shape
        
        # Generar matriz aleatoria
        omega = torch.randn(n, rank + 10, device=matrix.device, dtype=matrix.dtype)
        
        # Power iteration para mejor aproximación
        Y = matrix @ omega
        for _ in range(2):
            Y = matrix @ (matrix.T @ Y)
        
        # QR descomposición
        Q, _ = torch.linalg.qr(Y)
        
        # Proyectar y hacer SVD pequeño
        B = Q.T @ matrix
        U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde

        # Retornar solo los primeros 'rank' componentes. ``torch.linalg.svd``
        # devuelve ``Vh`` con forma (rank, n); usamos ``.t()`` para obtener una
        # matriz ``V`` de forma (n, rank) que coincida con la orientación de
        # ``torch.svd_lowrank``.
        return U[:, :rank], S[:rank], Vh[:rank, :].t()
        # Retornar solo los primeros 'rank' componentes.
        # ``torch.linalg.svd`` devuelve ``Vh`` con forma (rank, n), por lo que
        # lo transponemos para alinear con la convención de ``torch.svd_lowrank``
        # que retorna ``V`` con forma (n, rank).
        return U[:, :rank], S[:rank], Vh[:rank, :].T
      
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión por bajo rango"""
        if not isinstance(module, nn.Linear):
            return 0.0
        
        strength = config.get('strength', 0.5)
        rank_ratio = 1 - strength
        
        m, n = module.weight.shape
        original_params = m * n
        target_rank = int(min(m, n) * rank_ratio)
        compressed_params = target_rank * (m + n)
        
        return 1 - (compressed_params / original_params)

class AttentionPruning(CompressionMethod):
    """Poda de cabezas de atención optimizada"""
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Poda cabezas de atención menos importantes IN-PLACE"""
        # Verificar si es un módulo de atención
        if not self._is_attention_module(module):
            return module
        
        strength = config.get('strength', 0.5)
        
        # Buscar y modificar proyecciones de atención
        for name, child in module.named_children():
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj']):
                if isinstance(child, nn.Linear):
                    with torch.no_grad():
                        weight = child.weight.data
                        
                        # Estimar número de cabezas (GPT-2 usa 12 para small)
                        hidden_size = weight.shape[0]
                        # Intentar detectar num_heads
                        if hidden_size % 12 == 0:
                            num_heads = 12
                        elif hidden_size % 8 == 0:
                            num_heads = 8
                        elif hidden_size % 16 == 0:
                            num_heads = 16
                        else:
                            num_heads = 4  # Fallback
                        
                        head_dim = hidden_size // num_heads
                        
                        # Calcular importancia por cabeza
                        weight_reshaped = weight.reshape(num_heads, head_dim, -1)
                        head_importance = weight_reshaped.abs().mean(dim=(1, 2))
                        
                        # Seleccionar cabezas a mantener
                        num_keep = max(1, int(num_heads * (1 - strength)))
                        _, keep_indices = torch.topk(head_importance, num_keep)
                        keep_indices = keep_indices.sort()[0]
                        
                        # Crear máscara para las cabezas
                        mask = torch.zeros_like(weight)
                        for idx in keep_indices:
                            start = idx * head_dim
                            end = (idx + 1) * head_dim
                            mask[start:end, :] = 1
                        
                        # Aplicar máscara directamente al peso
                        child.weight.data *= mask
                        
                        logger.debug(f"Podadas {num_heads - num_keep}/{num_heads} cabezas en {name}")
        
        return module  # Devolver el mismo módulo modificado
    
    def _is_attention_module(self, module: nn.Module) -> bool:
        """Detecta si es un módulo de atención"""
        child_names = [name for name, _ in module.named_children()]
        attention_indicators = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'c_attn']
        return any(indicator in name for name in child_names for indicator in attention_indicators)
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión de atención"""
        strength = config.get('strength', 0.5)
        return strength * 0.7  # Las cabezas de atención típicamente comprimen menos

class MPOCompression(CompressionMethod):
    """Compresión por Matrix Product Operators usando TensorLy - NO crea módulos custom"""
    
    def __init__(self):
        if not TENSORLY_AVAILABLE:
            raise ImportError("TensorLy no está instalado. Instala con: pip install tensorly")
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica descomposición MPO IN-PLACE"""
        if not isinstance(module, nn.Linear):
            return module
        
        strength = config.get('strength', 0.5)
        
        with torch.no_grad():
            weight = module.weight.data
            original_shape = weight.shape
            original_dtype = weight.dtype
            
            # Configurar rango MPO basado en strength
            max_rank = min(original_shape[0], original_shape[1]) // 2
            mpo_rank = max(2, int(max_rank * (1 - strength)))
            
            try:
                # Convertir a float32 si es necesario
                if weight.dtype not in [torch.float32, torch.float64]:
                    weight = weight.float()
                
                # Factorizar dimensiones en pares
                factors_out = self._factorize_dimension(original_shape[0])
                factors_in = self._factorize_dimension(original_shape[1])
                
                # Reshape weight a formato 4D para MPO
                weight_4d = weight.reshape(factors_out[0], factors_out[1], 
                                         factors_in[0], factors_in[1])
                
                # Aplicar descomposición tensor_train
                factors = tensor_train(weight_4d, rank=mpo_rank)
                
                # Reconstruir tensor desde los factores
                weight_reconstructed = tl.tt_to_tensor(factors)
                
                # Reshape de vuelta a 2D
                weight_reconstructed = weight_reconstructed.reshape(original_shape)
                
                # Aplicar directamente al módulo original
                module.weight.data = weight_reconstructed.to(original_dtype)
                
                logger.debug(f"MPO aplicado: {original_shape} con rango {mpo_rank}")
                
            except Exception as e:
                logger.warning(f"Error en MPO, manteniendo peso original: {e}")
                # Si falla, intentar con SVD como fallback
                try:
                    svd = LowRankApproximation()
                    return svd.compress(module, config, device)
                except:
                    pass  # Mantener original si todo falla
        
        return module  # Devolver el mismo módulo modificado
    
    def _factorize_dimension(self, dim: int) -> Tuple[int, int]:
        """Factoriza una dimensión en dos factores cercanos"""
        sqrt_dim = int(np.sqrt(dim))
        for i in range(sqrt_dim, 0, -1):
            if dim % i == 0:
                return (i, dim // i)
        return (1, dim)
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión MPO"""
        if not isinstance(module, nn.Linear):
            return 0.0
        
        strength = config.get('strength', 0.5)
        # MPO típicamente logra compresión similar a SVD pero con mejor estructura
        return strength * 0.8

class TuckerDecomposition(CompressionMethod):
    """Descomposición Tucker real usando TensorLy - NO crea módulos custom"""
    
    def __init__(self):
        if not TENSORLY_AVAILABLE:
            logger.warning("TensorLy no disponible, Tucker usará SVD como aproximación")
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica descomposición Tucker IN-PLACE"""
        if not isinstance(module, nn.Linear):
            return module
        
        if not TENSORLY_AVAILABLE:
            # Fallback a SVD si no hay TensorLy
            svd = LowRankApproximation()
            return svd.compress(module, config, device)
        
        strength = config.get('strength', 0.5)
        
        with torch.no_grad():
            weight = module.weight.data
            original_shape = weight.shape
            original_dtype = weight.dtype
            
            try:
                # Convertir a float32 si es necesario
                if weight.dtype not in [torch.float32, torch.float64]:
                    weight = weight.float()
                
                # Tucker decomposition requiere tensor 3D o más
                # Para matriz 2D, agregamos dimensión dummy
                weight_3d = weight.unsqueeze(0)
                
                # Calcular ranks para Tucker
                rank_ratio = 1 - strength
                ranks = [
                    1,  # Dimensión dummy
                    max(1, int(weight.shape[0] * rank_ratio)),
                    max(1, int(weight.shape[1] * rank_ratio))
                ]
                
                # Aplicar Tucker decomposition
                from tensorly.decomposition import tucker
                core, factors = tucker(weight_3d, rank=ranks)
                
                # Reconstruir tensor
                weight_approx = tl.tucker_to_tensor((core, factors))
                weight_approx = weight_approx.squeeze(0)  # Quitar dimensión dummy
                
                # Aplicar directamente al módulo original
                module.weight.data = weight_approx.to(original_dtype)
                
                logger.debug(f"Tucker aplicado: {original_shape} con ranks {ranks[1:]}")
                
            except Exception as e:
                logger.warning(f"Error en Tucker, usando SVD como fallback: {e}")
                # Fallback a SVD
                svd = LowRankApproximation()
                return svd.compress(module, config, device)
        
        return module  # Devolver el mismo módulo modificado
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión Tucker"""
        if not isinstance(module, nn.Linear):
            return 0.0
        
        strength = config.get('strength', 0.5)
        rank_ratio = 1 - strength
        
        # Tucker típicamente comprime menos que SVD pero preserva más estructura
        m, n = module.weight.shape
        tucker_params = (m * n * rank_ratio**2) + (m + n) * rank_ratio
        original_params = m * n
        
        return 1 - (tucker_params / original_params)

class ExpertPruning(CompressionMethod):
    """Poda de expertos para modelos Mixture of Experts (MoE) - NO crea módulos custom"""
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica poda de expertos IN-PLACE si es un modelo MoE"""
        
        # Detectar si es un módulo MoE
        if not self._is_moe_module(module):
            logger.debug("Expert pruning: módulo no es MoE, retornando sin cambios")
            return module
        
        strength = config.get('strength', 0.5)
        
        # Buscar y modificar componentes de expertos
        if hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
            num_experts = len(module.experts)
            num_to_keep = max(1, int(num_experts * (1 - strength)))
            
            # Calcular importancia de cada experto
            expert_importance = []
            for i, expert in enumerate(module.experts):
                # Calcular importancia como norma total de pesos
                importance = 0
                for p in expert.parameters():
                    importance += p.abs().mean().item()
                expert_importance.append((i, importance))
            
            # Ordenar por importancia y seleccionar los que mantener
            expert_importance.sort(key=lambda x: x[1], reverse=True)
            keep_indices = set([idx for idx, _ in expert_importance[:num_to_keep]])
            
            # Poner a cero los expertos no seleccionados
            with torch.no_grad():
                for i, expert in enumerate(module.experts):
                    if i not in keep_indices:
                        # Poner todos los parámetros del experto a cero
                        for param in expert.parameters():
                            param.data.zero_()
                        logger.debug(f"Experto {i} eliminado (puesto a cero)")
            
            # Si hay un gate/router, ajustar sus pesos también
            if hasattr(module, 'gate') and hasattr(module.gate, 'weight'):
                with torch.no_grad():
                    # Poner a cero las salidas correspondientes a expertos eliminados
                    for i in range(num_experts):
                        if i not in keep_indices:
                            if i < module.gate.weight.shape[0]:
                                module.gate.weight.data[i].zero_()
                                if module.gate.bias is not None and i < module.gate.bias.shape[0]:
                                    module.gate.bias.data[i] = -float('inf')  # Desactivar efectivamente
            
            logger.info(f"Poda de expertos: {num_experts - num_to_keep}/{num_experts} expertos eliminados")
        
        # Buscar recursivamente en submódulos
        else:
            for name, child in module.named_children():
                if 'expert' in name.lower() or 'moe' in name.lower():
                    # Aplicar poda estructurada a expertos individuales
                    with torch.no_grad():
                        if hasattr(child, 'weight'):
                            # Aplicar poda por magnitud al experto
                            weight = child.weight.data
                            threshold = torch.quantile(weight.abs().flatten(), strength)
                            mask = weight.abs() > threshold
                            child.weight.data *= mask
                            logger.debug(f"Poda aplicada a experto: {name}")
        
        return module  # Devolver el mismo módulo modificado
    
    def _is_moe_module(self, module: nn.Module) -> bool:
        """Detecta si es un módulo MoE"""
        # Verificar por atributos típicos de MoE
        moe_indicators = ['experts', 'expert_weights', 'router', 'gate']
        
        # Verificar atributos directos
        for attr in moe_indicators:
            if hasattr(module, attr):
                return True
        
        # Verificar nombres de submódulos
        for name, _ in module.named_children():
            if any(indicator in name.lower() for indicator in ['expert', 'moe', 'mixture']):
                return True
        
        # Verificar tipo de clase
        module_class = module.__class__.__name__.lower()
        if any(term in module_class for term in ['moe', 'mixture', 'expert']):
            return True
        
        return False
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión de expertos"""
        if not self._is_moe_module(module):
            return 0.0
        
        strength = config.get('strength', 0.5)
        # La poda de expertos es muy efectiva
        return strength * 0.95

class SVDDecomposition(CompressionMethod):
    """Descomposición SVD explícita - NO crea módulos custom"""
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica descomposición SVD IN-PLACE"""
        # Usar LowRankApproximation que ya implementa SVD correctamente
        svd = LowRankApproximation()
        return svd.compress(module, config, device)
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión SVD"""
        svd = LowRankApproximation()
        return svd.estimate_compression(module, config)

class MixedPrecisionMethod(CompressionMethod):
    """Precisión mixta adaptativa - NO crea módulos custom"""
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica precisión mixta IN-PLACE"""
        if not hasattr(module, 'weight'):
            return module
        
        strength = config.get('strength', 0.5)
        
        with torch.no_grad():
            weight = module.weight.data
            original_dtype = weight.dtype
            
            # Calcular importancia de cada peso
            importance = weight.abs()
            
            # Dividir pesos en regiones por importancia
            threshold_high = torch.quantile(importance.flatten(), 1 - strength * 0.3)
            threshold_low = torch.quantile(importance.flatten(), strength * 0.5)
            
            # Crear máscaras para diferentes precisiones
            mask_high = importance > threshold_high  # Alta precisión (sin cambios)
            mask_low = importance < threshold_low    # Baja precisión (más cuantización)
            mask_medium = ~mask_high & ~mask_low     # Precisión media
            
            # Aplicar diferentes niveles de cuantización
            weight_mixed = weight.clone()
            
            # Región de baja importancia: cuantización agresiva (simular INT4)
            if mask_low.any():
                low_weights = weight[mask_low]
                # Cuantizar a 4 bits
                scale = (low_weights.max() - low_weights.min()) / 15  # 2^4 - 1 = 15
                if scale > 0:
                    quantized = torch.round(low_weights / scale) * scale
                    weight_mixed[mask_low] = quantized
            
            # Región media: cuantización moderada (simular INT8)
            if mask_medium.any():
                medium_weights = weight[mask_medium]
                # Cuantizar a 8 bits
                scale = (medium_weights.max() - medium_weights.min()) / 255  # 2^8 - 1 = 255
                if scale > 0:
                    quantized = torch.round(medium_weights / scale) * scale
                    weight_mixed[mask_medium] = quantized
            
            # Región alta: mantener precisión original (mask_high no se toca)
            
            # Aplicar directamente al módulo
            module.weight.data = weight_mixed.to(original_dtype)
            
            logger.debug(f"Precisión mixta aplicada: {(mask_high.sum()/weight.numel()*100):.1f}% alta, "
                        f"{(mask_medium.sum()/weight.numel()*100):.1f}% media, "
                        f"{(mask_low.sum()/weight.numel()*100):.1f}% baja")
        
        return module  # Devolver el mismo módulo modificado
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión por precisión mixta"""
        strength = config.get('strength', 0.5)
        # Aproximación: 30% mantiene precisión completa, resto se reduce
        high_precision_ratio = 1 - strength * 0.3
        medium_precision_ratio = strength * 0.5
        low_precision_ratio = strength * 0.2
        
        # Calcular compresión promedio ponderada
        compression = (high_precision_ratio * 0 +      # Sin compresión
                      medium_precision_ratio * 0.5 +   # 50% compresión (INT8)
                      low_precision_ratio * 0.75)      # 75% compresión (INT4)
        
        return compression

class BlockSparseMethod(CompressionMethod):
    """Sparsidad por bloques - NO crea módulos custom"""
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        """Aplica sparsidad por bloques IN-PLACE"""
        if not hasattr(module, 'weight'):
            return module
        
        strength = config.get('strength', 0.5)
        block_size = config.get('block_size', 16)
        
        with torch.no_grad():
            weight = module.weight.data
            original_dtype = weight.dtype
            
            # Convertir a float32 si es necesario para cálculos
            if weight.dtype not in [torch.float32, torch.float64]:
                weight_float = weight.float()
            else:
                weight_float = weight
            
            # Dividir en bloques y calcular importancia
            h, w = weight.shape
            h_blocks = (h + block_size - 1) // block_size
            w_blocks = (w + block_size - 1) // block_size
            
            # Calcular importancia por bloque
            block_importance = torch.zeros(h_blocks, w_blocks, device=weight.device)
            
            for i in range(h_blocks):
                for j in range(w_blocks):
                    h_start = i * block_size
                    h_end = min((i + 1) * block_size, h)
                    w_start = j * block_size
                    w_end = min((j + 1) * block_size, w)
                    
                    block = weight_float[h_start:h_end, w_start:w_end]
                    block_importance[i, j] = block.abs().mean()
            
            # Seleccionar bloques a mantener
            threshold = torch.quantile(block_importance.flatten(), strength)
            
            # Crear máscara de peso completo
            mask = torch.ones_like(weight, dtype=torch.bool)
            
            # Aplicar máscara por bloques
            blocks_zeroed = 0
            total_blocks = h_blocks * w_blocks
            
            for i in range(h_blocks):
                for j in range(w_blocks):
                    if block_importance[i, j] <= threshold:
                        h_start = i * block_size
                        h_end = min((i + 1) * block_size, h)
                        w_start = j * block_size
                        w_end = min((j + 1) * block_size, w)
                        
                        mask[h_start:h_end, w_start:w_end] = False
                        blocks_zeroed += 1
            
            # Aplicar máscara directamente al peso
            module.weight.data = weight * mask.to(original_dtype)
            
            logger.debug(f"Sparsidad por bloques aplicada: {blocks_zeroed}/{total_blocks} bloques "
                        f"({blocks_zeroed/total_blocks*100:.1f}%) eliminados, tamaño de bloque: {block_size}")
        
        return module  # Devolver el mismo módulo modificado
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        """Estima compresión por bloques"""
        strength = config.get('strength', 0.5)
        # Los bloques típicamente comprimen un poco menos que poda individual
        return strength * 0.9

# Funciones getter para todos los métodos
def get_int8_quantization():
    return QuantizationMethod(bits=8)

def get_int4_quantization():
    return QuantizationMethod(bits=4)

def get_int2_quantization():
    return QuantizationMethod(bits=2)

def get_magnitude_pruning():
    return PruningMethod(structured=False)

def get_structured_pruning():
    return PruningMethod(structured=True)

def get_low_rank_approximation():
    return LowRankApproximation()

def get_attention_pruning():
    return AttentionPruning()

def get_head_pruning():
    return AttentionPruning()

def get_tensor_decomposition():
    return LowRankApproximation()

def get_mpo():
    return MPOCompression()

def get_tucker():
    return TuckerDecomposition()

def get_svd():
    return SVDDecomposition()

def get_mixed_precision():
    return MixedPrecisionMethod()

def get_block_sparse():
    return BlockSparseMethod()

def get_expert_pruning():
    return ExpertPruning()

def get_none():
    return NullCompression()

# Diccionario completo de métodos (sin los que requieren entrenamiento)
COMPRESSION_METHODS = {
    'int8_quantization': get_int8_quantization,
    'int4_quantization': get_int4_quantization,
    'int2_quantization': get_int2_quantization,
    'magnitude_pruning': get_magnitude_pruning,
    'structured_pruning': get_structured_pruning,
    'low_rank_approximation': get_low_rank_approximation,
    'attention_pruning': get_attention_pruning,
    'head_pruning': get_head_pruning,
    'tensor_decomposition': get_tensor_decomposition,
    'mpo': get_mpo,
    'tucker': get_tucker,
    'svd': get_svd,
    'mixed_precision': get_mixed_precision,
    'block_sparse': get_block_sparse,
    'expert_pruning': get_expert_pruning,
    'none': get_none
}

class NullCompression(CompressionMethod):
    """No aplicar compresión"""
    
    def compress(self, module: nn.Module, config: Dict[str, Any], 
                device: torch.device) -> nn.Module:
        return module
    
    def estimate_compression(self, module: nn.Module, config: Dict[str, Any]) -> float:
        return 0.0

def get_compression_method(name: str) -> CompressionMethod:
    """Obtiene método de compresión con cache"""
    if name not in _COMPILED_METHODS_CACHE:
        if name not in COMPRESSION_METHODS:
            logger.warning(f"Método desconocido: {name}, usando 'none'")
            name = 'none'
        
        # Crear instancia y cachear
        method = COMPRESSION_METHODS[name]()
        _COMPILED_METHODS_CACHE[name] = method
    
    return _COMPILED_METHODS_CACHE[name]

def apply_compression(module: nn.Module, method_config: Dict[str, Any], 
                     device: torch.device) -> nn.Module:
    """Aplica un método de compresión a un módulo"""
    method_name = method_config.get('name', 'none')
    method = get_compression_method(method_name)
    
    try:
        return method.compress(module, method_config, device)
    except Exception as e:
        logger.error(f"Error aplicando {method_name}: {e}")
        return module

def estimate_compression_ratio(module: nn.Module, methods: List[Dict[str, Any]]) -> float:
    """Estima ratio de compresión total para múltiples métodos"""
    if not methods:
        return 0.0
    
    # Los métodos se aplican secuencialmente
    total_compression = 1.0
    
    for method_config in methods:
        method_name = method_config.get('name', 'none')
        method = get_compression_method(method_name)
        
        # Compresión de este método
        compression = method.estimate_compression(module, method_config)
        
        # Aplicar de forma multiplicativa
        total_compression *= (1 - compression)
    
    return 1 - total_compression

# Funciones auxiliares optimizadas
@lru_cache(maxsize=100)
def get_available_methods() -> Dict[str, str]:
    """Retorna métodos disponibles con descripciones"""
    methods = {
        'int8_quantization': 'Cuantización a 8 bits - Balance entre compresión y precisión',
        'int4_quantization': 'Cuantización a 4 bits - Mayor compresión, menor precisión',
        'int2_quantization': 'Cuantización a 2 bits - Máxima compresión, mínima precisión',
        'magnitude_pruning': 'Poda por magnitud - Elimina pesos pequeños',
        'structured_pruning': 'Poda estructurada - Elimina neuronas/canales completos',
        'low_rank_approximation': 'Aproximación de bajo rango - Factorización matricial SVD',
        'attention_pruning': 'Poda de atención - Elimina cabezas menos importantes',
        'head_pruning': 'Poda de cabezas - Alias de attention_pruning',
        'tensor_decomposition': 'Descomposición tensorial - Alias de low_rank',
        'svd': 'Descomposición SVD - Singular Value Decomposition explícita',
        'mixed_precision': 'Precisión mixta - Diferentes precisiones por importancia',
        'block_sparse': 'Sparsidad por bloques - Elimina bloques de pesos',
        'expert_pruning': 'Poda de expertos - Solo para modelos Mixture of Experts',
        'none': 'Sin compresión - Mantener capa original'
    }
    
    # Agregar métodos avanzados si TensorLy está disponible
    if TENSORLY_AVAILABLE:
        methods.update({
            'mpo': 'Matrix Product Operators - Descomposición tensorial avanzada',
            'tucker': 'Descomposición Tucker - Factorización multidimensional',
        })
    else:
        methods.update({
            'mpo': '[NO DISPONIBLE - Instalar TensorLy] Matrix Product Operators',
            'tucker': '[NO DISPONIBLE - Instalar TensorLy] Descomposición Tucker',
        })
    
    return methods

def validate_compression_config(config: Dict[str, Any]) -> bool:
    """Valida configuración de compresión"""
    required_fields = ['name']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Campo requerido faltante: {field}")
            return False
    
    # Validar método existe
    if config['name'] not in COMPRESSION_METHODS:
        logger.error(f"Método desconocido: {config['name']}")
        return False
    
    # Validar strength
    strength = config.get('strength', 0.5)
    if not 0 <= strength <= 1:
        logger.error(f"Strength debe estar entre 0 y 1: {strength}")
        return False
    
    # Advertencia especial para métodos que requieren TensorLy
    if config['name'] in ['mpo', 'tucker'] and not TENSORLY_AVAILABLE:
        logger.warning(f"{config['name']} requiere TensorLy. Instala con: pip install tensorly")
    
    return True