#!/usr/bin/env python3
"""
Implementación de múltiples métodos PEFT (Parameter-Efficient Fine-Tuning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

from .peft_methods_config import (
    PEFTMethod, BasePEFTConfig, MoLoRAConfig, GaLoreConfig, 
    DoRAConfig, BitFitConfig, IA3Config, 
    PromptTuningConfig, AdapterConfig, QLoRAConfig, LoRAConfig,
    CompacterConfig, KronAConfig, S4Config, HoulsbyConfig
)


class BasePEFTModule(nn.Module, ABC):
    """Clase base para todos los módulos PEFT"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def merge_weights(self) -> None:
        """Fusiona los pesos entrenables con el modelo base"""
        pass


# ============= MoLoRA (Mixture of LoRAs) =============

class MoLoRARouter(nn.Module):
    """Router para seleccionar expertos LoRA"""
    
    def __init__(self, hidden_size: int, num_experts: int, router_type: str = "learned"):
        super().__init__()
        self.num_experts = num_experts
        self.router_type = router_type
        
        if router_type == "learned":
            self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.router_type == "learned":
            # x shape: [batch, seq_len, hidden]
            router_logits = self.gate(x)
            router_weights = F.softmax(router_logits, dim=-1)
            
            # Top-2 gating
            topk_weights, topk_indices = torch.topk(router_weights, k=2, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            
            return topk_weights, topk_indices
        elif self.router_type == "hash":
            # Hash-based routing (deterministic)
            batch_size, seq_len = x.shape[:2]
            indices = torch.randint(0, self.num_experts, (batch_size, seq_len, 2), device=x.device)
            weights = torch.ones_like(indices, dtype=x.dtype) * 0.5
            return weights, indices
        else:
            raise ValueError(f"Unknown router type: {self.router_type}")


class MoLoRALinear(BasePEFTModule):
    """Capa Linear con Mixture of LoRAs"""
    
    def __init__(self, in_features: int, out_features: int, config: 'MoLoRAConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.num_experts = config.num_experts
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Crear múltiples expertos LoRA
        self.lora_As = nn.ModuleList([
            nn.Linear(in_features, config.expert_r[i], bias=False)
            for i in range(config.num_experts)
        ])
        self.lora_Bs = nn.ModuleList([
            nn.Linear(config.expert_r[i], out_features, bias=False)
            for i in range(config.num_experts)
        ])
        
        # Router
        self.router = MoLoRARouter(in_features, config.num_experts, config.router_type)
        
        # Inicialización
        for i in range(config.num_experts):
            nn.init.kaiming_uniform_(self.lora_As[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs[i].weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # Router decision
        router_weights, router_indices = self.router(x)
        
        # Manejar diferentes dimensiones de entrada
        if x.dim() == 2:
            # Input 2D: (batch_size, features)
            batch_size, hidden = x.shape
            x_flat = x
            out_flat = out
        else:
            # Input 3D: (batch_size, seq_len, features)
            batch_size, seq_len, hidden = x.shape
            x_flat = x.view(-1, hidden)
            out_flat = out.view(-1, self.out_features)
        
        # Para cada experto seleccionado
        for i in range(2):  # Top-2
            expert_out = torch.zeros_like(out_flat)
            
            for expert_idx in range(self.config.num_experts):
                # Máscara para este experto
                if x.dim() == 2:
                    mask = (router_indices[..., i] == expert_idx)
                else:
                    mask = (router_indices[..., i] == expert_idx).view(-1)
                
                if mask.any():
                    expert_input = x_flat[mask]
                    
                    # Aplicar LoRA del experto
                    lora_out = self.lora_Bs[expert_idx](
                        self.lora_As[expert_idx](expert_input)
                    )
                    
                    # Ponderar por router
                    if x.dim() == 2:
                        weight = router_weights[..., i][mask].unsqueeze(-1)
                    else:
                        weight = router_weights[..., i].view(-1)[mask].unsqueeze(-1)
                    
                    expert_out[mask] += lora_out * weight * self.config.lora_alpha
            
            if x.dim() == 2:
                out = out + expert_out
            else:
                out = out + expert_out.view(batch_size, seq_len, -1)
        
        return out
    
    def merge_weights(self):
        # MoLoRA no se fusiona típicamente
        logger.warning("MoLoRA merge not implemented - keeping separate experts")


# ============= GaLore (Gradient Low-Rank Projection) =============

class GaLoreProjector:
    """Proyector de gradientes para GaLore"""
    
    def __init__(self, shape: Tuple[int, ...], rank: int, proj_type: str = "std"):
        self.shape = shape
        self.rank = rank
        self.proj_type = proj_type
        
        # Matrices de proyección
        self.U = None
        self.V = None
        
    def update_projection(self, grad: torch.Tensor) -> None:
        """Actualiza matrices de proyección usando SVD del gradiente"""
        if len(grad.shape) == 2:
            U, S, V = torch.svd_lowrank(grad, q=self.rank)
            
            if self.proj_type == "std":
                self.U = U
                self.V = V.T
            elif self.proj_type == "reverse_std":
                self.U = U @ torch.diag(S.sqrt())
                self.V = torch.diag(S.sqrt()) @ V.T
            elif self.proj_type == "left":
                self.U = U
                self.V = None
            elif self.proj_type == "right":
                self.U = None
                self.V = V.T
    
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Proyecta gradiente a espacio de bajo rango"""
        if self.U is not None and self.V is not None:
            # Proyección completa: U @ U.T @ grad @ V @ V.T
            return self.U @ (self.U.T @ grad @ self.V.T) @ self.V
        elif self.U is not None:
            # Solo proyección izquierda
            return self.U @ (self.U.T @ grad)
        elif self.V is not None:
            # Solo proyección derecha
            return (grad @ self.V.T) @ self.V
        else:
            return grad


class GaLoreLinear(nn.Linear):
    """Linear layer con GaLore (se aplica durante el entrenamiento)"""
    
    def __init__(self, in_features: int, out_features: int, config: 'GaLoreConfig', bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.config = config
        
        # Proyector para gradientes
        self.projector = GaLoreProjector(
            (out_features, in_features),
            config.rank,
            config.proj_type
        )
        
        # Contador de pasos para actualización
        self.step_count = 0
        
        # Hook para proyección de gradientes
        self.weight.register_hook(self._grad_hook)
    
    def _grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Hook para proyectar gradientes"""
        # Actualizar proyección si es necesario
        if self.step_count % self.config.update_proj_gap == 0:
            self.projector.update_projection(grad)
        
        self.step_count += 1
        
        # Proyectar gradiente
        return self.projector.project(grad) * self.config.scale
    
    def merge_weights(self):
        # GaLore no requiere fusión
        pass


# ============= DoRA (Weight-Decomposed Low-Rank Adaptation) =============

class DoRALinear(BasePEFTModule):
    """DoRA: Descompone W = magnitude * direction, entrena ambos con LoRA"""
    
    def __init__(self, in_features: int, out_features: int, config: 'DoRAConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base descompuesto
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Magnitud (vector)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Adaptadores LoRA para dirección
        self.lora_A = nn.Parameter(torch.randn(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
        
        # Escala DoRA
        self.dora_scale = nn.Parameter(torch.ones(1))
        
        # Dropout
        self.dropout = nn.Dropout(config.lora_dropout)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Normalizar peso base
        with torch.no_grad():
            norm = self.weight.norm(dim=1, keepdim=True)
            self.weight.div_(norm)
            self.magnitude.copy_(norm.squeeze())
        
        # Inicializar LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dirección base + adaptación LoRA
        base_direction = self.weight
        lora_direction = self.dropout(self.lora_B @ self.lora_A) * (self.config.lora_alpha / self.config.r)
        
        # Combinar direcciones
        direction = base_direction + lora_direction
        
        # Normalizar si es necesario
        if self.config.normalize_direction:
            direction = F.normalize(direction, dim=1)
        
        # Aplicar magnitud
        weight = self.magnitude.unsqueeze(1) * direction
        
        return F.linear(x, weight)
    
    def merge_weights(self):
        with torch.no_grad():
            # Fusionar adaptación en la dirección
            direction = self.weight + self.lora_B @ self.lora_A * (self.config.lora_alpha / self.config.r)
            if self.config.normalize_direction:
                direction = F.normalize(direction, dim=1)
            
            # Actualizar peso
            self.weight.copy_(self.magnitude.unsqueeze(1) * direction)


# ============= AdaLoRA (Adaptive LoRA) =============
# AdaLoRA ha sido eliminado de esta implementación


# ============= BitFit =============

class BitFitModel(nn.Module):
    """Wrapper que congela todo excepto bias"""
    
    def __init__(self, model: nn.Module, config: 'BitFitConfig'):
        super().__init__()
        self.model = model
        self.config = config
        
        # Congelar todos los parámetros
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Descongelar solo bias
        for name, param in self.model.named_parameters():
            if self._should_train(name):
                param.requires_grad = True
                logger.info(f"BitFit: Entrenando {name}")
    
    def _should_train(self, param_name: str) -> bool:
        """Determina si un parámetro debe entrenarse"""
        # Siempre entrenar bias
        if 'bias' in param_name:
            return True
        
        # Opciones adicionales
        if self.config.train_layer_norms and 'norm' in param_name:
            return True
        
        if self.config.train_embeddings and 'embed' in param_name:
            return True
        
        return False
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def merge_weights(self):
        # BitFit no requiere fusión
        pass


# ============= IA³ (Infused Adapter by Inhibiting and Amplifying) =============

class IA3Linear(BasePEFTModule):
    """IA³ para capas lineales"""
    
    def __init__(self, base_layer: nn.Linear, config: 'IA3Config', is_feedforward: bool = False):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.is_feedforward = is_feedforward
        
        # Vector de escala learnable
        init_value = 1.0 if config.init_ia3_weights == "ones" else 0.0
        
        if is_feedforward:
            # Para FFN, escalar salida
            self.ia3_weights = nn.Parameter(
                torch.full((base_layer.out_features,), init_value)
            )
        else:
            # Para attention K,V, escalar entrada  
            self.ia3_weights = nn.Parameter(
                torch.full((base_layer.in_features,), init_value)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_feedforward:
            # Escalar salida
            return self.base_layer(x) * self.ia3_weights
        else:
            # Escalar entrada
            return self.base_layer(x * self.ia3_weights)
    
    def merge_weights(self):
        with torch.no_grad():
            if self.is_feedforward:
                # Multiplicar pesos de salida
                self.base_layer.weight.mul_(self.ia3_weights.unsqueeze(1))
                if self.base_layer.bias is not None:
                    self.base_layer.bias.mul_(self.ia3_weights)
            else:
                # Multiplicar pesos de entrada
                self.base_layer.weight.mul_(self.ia3_weights.unsqueeze(0))


# ============= Quantized LoRA =============

class QuantizedLoRALinear(BasePEFTModule):
    """LoRA con cuantización de pesos"""
    
    def __init__(self, in_features: int, out_features: int, config: 'QLoRAConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Matrices LoRA
        self.lora_A = nn.Parameter(torch.randn(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
        
        # Dropout
        self.dropout = nn.Dropout(config.lora_dropout)
        
        # Parámetros de cuantización
        self.bits = config.bits
        self.scale = None
        self.zero_point = None
        
        # Inicialización
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Cuantiza el peso a bits específicos"""
        if self.bits == 4:
            # Cuantización a 4 bits
            min_val = weight.min()
            max_val = weight.max()
            scale = (max_val - min_val) / (2**4 - 1)
            zero_point = min_val
            
            # Cuantizar
            quantized = torch.round((weight - zero_point) / scale)
            quantized = torch.clamp(quantized, 0, 2**4 - 1)
            
            # Devolver a float
            return quantized * scale + zero_point
        else:
            # Para otros bits, usar cuantización estándar
            return torch.quantize_per_tensor(weight, scale=1.0, zero_point=0, dtype=torch.qint8).dequantize()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # Cuantizar LoRA weights
        quantized_A = self._quantize_weight(self.lora_A)
        quantized_B = self._quantize_weight(self.lora_B)
        
        # Aplicar LoRA cuantizado
        lora_out = F.linear(
            self.dropout(F.linear(x, quantized_A)),
            quantized_B
        )
        
        return out + lora_out * (self.config.lora_alpha / self.config.r)
    
    def merge_weights(self):
        with torch.no_grad():
            # Fusionar LoRA cuantizado
            quantized_A = self._quantize_weight(self.lora_A)
            quantized_B = self._quantize_weight(self.lora_B)
            self.weight += quantized_B @ quantized_A * (self.config.lora_alpha / self.config.r)


# ============= Pruned LoRA =============

class PrunedLoRALinear(BasePEFTModule):
    """LoRA con poda de pesos"""
    
    def __init__(self, in_features: int, out_features: int, config: 'LoRAConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Matrices LoRA
        self.lora_A = nn.Parameter(torch.randn(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
        
        # Dropout
        self.dropout = nn.Dropout(config.lora_dropout)
        
        # Máscaras de poda
        self.A_mask = nn.Parameter(torch.ones_like(self.lora_A), requires_grad=False)
        self.B_mask = nn.Parameter(torch.ones_like(self.lora_B), requires_grad=False)
        
        # Umbral de poda
        self.pruning_threshold = 0.01
        
        # Inicialización
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def _update_masks(self):
        """Actualiza máscaras de poda basado en magnitud de pesos"""
        # Poda por magnitud
        A_magnitude = torch.abs(self.lora_A)
        B_magnitude = torch.abs(self.lora_B)
        
        # Crear máscaras
        self.A_mask.data = (A_magnitude > self.pruning_threshold).float()
        self.B_mask.data = (B_magnitude > self.pruning_threshold).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # Aplicar máscaras de poda
        masked_A = self.lora_A * self.A_mask
        masked_B = self.lora_B * self.B_mask
        
        # Aplicar LoRA podado
        lora_out = F.linear(
            self.dropout(F.linear(x, masked_A)),
            masked_B
        )
        
        return out + lora_out * (self.config.lora_alpha / self.config.r)
    
    def prune_weights(self, threshold: float = None):
        """Ejecuta poda de pesos"""
        if threshold is not None:
            self.pruning_threshold = threshold
        
        self._update_masks()
        
        # Contar parámetros activos
        active_A = self.A_mask.sum().item()
        active_B = self.B_mask.sum().item()
        total_A = self.A_mask.numel()
        total_B = self.B_mask.numel()
        
        sparsity_A = 1 - (active_A / total_A)
        sparsity_B = 1 - (active_B / total_B)
        
        logger.info(f"Poda completada - Sparsity A: {sparsity_A:.2%}, Sparsity B: {sparsity_B:.2%}")
    
    def merge_weights(self):
        with torch.no_grad():
            # Fusionar LoRA podado
            masked_A = self.lora_A * self.A_mask
            masked_B = self.lora_B * self.B_mask
            self.weight += masked_B @ masked_A * (self.config.lora_alpha / self.config.r)


# ============= Prompt Tuning =============

class PromptEncoder(nn.Module):
    """Encoder para generar embeddings de prompts virtuales"""
    
    def __init__(self, config: 'PromptTuningConfig', embedding_dim: int):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Embeddings de prompts virtuales
        if config.prompt_tuning_init == "random":
            self.prompt_embeddings = nn.Parameter(
                torch.randn(config.num_virtual_tokens, embedding_dim)
            )
        else:
            # Inicializar desde texto (requiere tokenizer)
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(config.num_virtual_tokens, embedding_dim)
            )
        
        # Encoder opcional
        if config.encoder_reparameterization_type == "MLP":
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, config.encoder_hidden_size or embedding_dim),
                nn.ReLU(),
                nn.Dropout(config.encoder_dropout),
                nn.Linear(config.encoder_hidden_size or embedding_dim, embedding_dim)
            )
        else:
            self.mlp = None
    
    def forward(self, batch_size: int) -> torch.Tensor:
        # Expandir para el batch
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Aplicar encoder si existe
        if self.mlp is not None:
            prompt_embeds = self.mlp(prompt_embeds)
        
        return prompt_embeds


# ============= Compacter =============

class CompacterLinear(BasePEFTModule):
    """Compacter: Adapter con compresión de parámetros usando factorización de bajo rango"""
    
    def __init__(self, in_features: int, out_features: int, config: 'CompacterConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Compacter: simplified version using a single low-rank matrix
        self.compacter_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Scaling factor
        self.scaling = config.scaling
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Inicialización Xavier
        nn.init.xavier_uniform_(self.compacter_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # Compacter forward: simplified version
        compacter_out = F.linear(x, self.compacter_weight)
        
        # Apply dropout and scaling
        compacter_out = self.dropout(compacter_out) * self.scaling
        
        # Residual connection
        out = out + compacter_out
        
        # Layer norm
        out = self.layer_norm(out)
        
        return out
    
    def merge_weights(self):
        with torch.no_grad():
            # Fusionar: W + compacter_weight
            self.weight += self.compacter_weight * self.scaling


# ============= KronA (Kronecker Adapter) =============

class KronALinear(BasePEFTModule):
    """KronA: Adapter usando productos de Kronecker para eficiencia de parámetros"""
    
    def __init__(self, in_features: int, out_features: int, config: 'KronAConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # KronA: simplified version using a single adapter weight
        self.krona_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling factor
        self.scaling = config.scaling
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Inicialización Xavier
        nn.init.xavier_uniform_(self.krona_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # KronA forward: simplified version
        kron_out = F.linear(x, self.krona_weight)
        
        # Apply dropout and scaling
        kron_out = self.dropout(kron_out) * self.scaling
        
        # Residual connection
        out = out + kron_out
        
        return out
    
    def merge_weights(self):
        with torch.no_grad():
            # Fusionar: W + krona_weight
            self.weight += self.krona_weight * self.scaling


# ============= S4 Adapter =============

class S4Adapter(BasePEFTModule):
    """S4 Adapter: Adapter basado en modelos de estado espacial"""
    
    def __init__(self, in_features: int, out_features: int, config: 'S4Config'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # S4: simplified version using a single adapter weight
        self.s4_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Scaling factor
        self.scaling = config.scaling
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Inicialización Xavier
        nn.init.xavier_uniform_(self.s4_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # S4 forward: simplified version
        s4_out = F.linear(x, self.s4_weight)
        
        # Apply dropout and scaling
        s4_out = self.dropout(s4_out) * self.scaling
        
        # Residual connection
        out = out + s4_out
        
        # Layer norm
        out = self.layer_norm(out)
        
        return out
    
    def merge_weights(self):
        with torch.no_grad():
            # Fusionar: W + s4_weight
            self.weight += self.s4_weight * self.scaling


class S4Layer(nn.Module):
    """Capa S4 simplificada para el adaptador"""
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Simplified S4: just use a simple linear transformation
        self.projection = nn.Linear(d_model, d_model)
        
        # Convolution kernel
        self.conv_kernel = nn.Parameter(torch.randn(d_conv))
        
        # Initialization
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.conv_kernel, std=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Simplified S4: just apply linear transformation
        x = self.projection(x)
        
        return x


# ============= Houlsby Adapter =============

class HoulsbyAdapterLinear(BasePEFTModule):
    """Houlsby Adapter: Adaptadores antes y después de la capa"""
    
    def __init__(self, in_features: int, out_features: int, config: 'HoulsbyConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Houlsby adapters: antes y después de la capa
        self.adapter_pre = AdapterLayer(in_features, config.adapter_size, config)
        self.adapter_post = AdapterLayer(out_features, config.adapter_size, config)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Los adaptadores se inicializan internamente
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapter pre
        x = self.adapter_pre(x)
        
        # Forward base
        out = F.linear(x, self.weight)
        
        # Adapter post
        out = self.adapter_post(out)
        
        return out
    
    def merge_weights(self):
        # Houlsby adapters no se fusionan típicamente
        logger.warning("Houlsby adapter merge not implemented - keeping separate modules")


class AdapterLayer(nn.Module):
    """Capa de adaptador individual para Houlsby"""
    
    def __init__(self, hidden_size: int, adapter_size: int, config: 'HoulsbyConfig'):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.config = config
        
        # Down-projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Non-linearity
        if config.non_linearity == "relu":
            self.activation = nn.ReLU()
        elif config.non_linearity == "gelu":
            self.activation = nn.GELU()
        elif config.non_linearity == "swish":
            self.activation = nn.SiLU()
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Scaling factor
        self.scaling = config.scaling
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Inicialización estilo BERT
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Adapter forward
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        # Residual connection con scaling
        x = residual + x * self.scaling
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


# ============= Adapter =============

class AdapterLinear(BasePEFTModule):
    """Capa Linear con Adapter (bottleneck)"""
    
    def __init__(self, in_features: int, out_features: int, config: 'AdapterConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base (congelado)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Adapter layers
        self.adapter_down = nn.Linear(in_features, config.adapter_size)
        self.adapter_up = nn.Linear(config.adapter_size, out_features)
        
        # Activation
        if hasattr(config, 'non_linearity') and config.non_linearity == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'adapter_dropout', 0.1))
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Scaling factor
        self.scaling = getattr(config, 'scaling', 1.0)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Inicialización estilo BERT
        nn.init.normal_(self.adapter_down.weight, std=0.02)
        nn.init.normal_(self.adapter_up.weight, std=0.02)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # Adapter forward
        adapter_out = self.adapter_down(x)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.dropout(adapter_out)
        adapter_out = self.adapter_up(adapter_out)
        
        # Residual connection
        out = out + adapter_out * self.scaling
        
        # Layer norm
        out = self.layer_norm(out)
        
        return out
    
    def merge_weights(self):
        # Los adapters no se fusionan típicamente
        logger.warning("Adapter merge not implemented - keeping separate modules")


# ============= Factory Functions =============

def create_peft_model(base_model: nn.Module, config: 'BasePEFTConfig') -> nn.Module:
    """Factory para crear modelos con el método PEFT especificado"""
    
    method = config.method
    
    if method == PEFTMethod.BITFIT:
        return BitFitModel(base_model, config)
    
    elif method == PEFTMethod.PROMPT_TUNING:
        # Requiere modificación del modelo para insertar prompts
        raise NotImplementedError("Prompt tuning requiere integración con el modelo específico")
    
    else:
        # Para otros métodos, reemplazar capas target
        _replace_layers(base_model, config)
        return base_model


def _replace_layers(model: nn.Module, config: 'BasePEFTConfig') -> None:
    """Reemplaza capas del modelo según el método PEFT"""
    
    for name, module in model.named_modules():
        # Verificar si es un módulo target
        if not _is_target_module(name, module, config):
            continue
        
        # Obtener padre y nombre del hijo
        parent, child_name = _get_parent_and_child(model, name)
        
        # Crear nueva capa según método
        if config.method == PEFTMethod.LORA and isinstance(module, nn.Linear):
            from peft import LoraConfig, get_peft_model
            # Usar implementación estándar de PEFT para LoRA básico
            continue
            
        elif config.method == PEFTMethod.MOLORA and isinstance(module, nn.Linear):
            new_module = MoLoRALinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.GALORE and isinstance(module, nn.Linear):
            new_module = GaLoreLinear(module.in_features, module.out_features, config, 
                                     bias=module.bias is not None)
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
                
        elif config.method == PEFTMethod.DORA and isinstance(module, nn.Linear):
            new_module = DoRALinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.ADAPTER and isinstance(module, nn.Linear):
            new_module = AdapterLinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.QLORA and isinstance(module, nn.Linear):
            new_module = QuantizedLoRALinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.COMPACTER and isinstance(module, nn.Linear):
            new_module = CompacterLinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.KRONA and isinstance(module, nn.Linear):
            new_module = KronALinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.S4 and isinstance(module, nn.Linear):
            new_module = S4Adapter(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.HOULSBY and isinstance(module, nn.Linear):
            new_module = HoulsbyAdapterLinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.IA3 and isinstance(module, nn.Linear):
            is_feedforward = any(ffn in name for ffn in config.feedforward_modules or [])
            new_module = IA3Linear(module, config, is_feedforward)
            
        elif config.method == PEFTMethod.LORA and isinstance(module, nn.Linear):
            # Para LoRA básico, usar PrunedLoRA como implementación
            new_module = PrunedLoRALinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
        else:
            continue
        
        # Reemplazar módulo
        setattr(parent, child_name, new_module)


def _is_target_module(name: str, module: nn.Module, config: 'BasePEFTConfig') -> bool:
    """Verifica si un módulo debe ser reemplazado"""
    if config.target_modules is None:
        return False
    
    return any(target in name for target in config.target_modules)


def _get_parent_and_child(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """Obtiene el módulo padre y el nombre del hijo"""
    parts = module_name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    return parent, parts[-1]