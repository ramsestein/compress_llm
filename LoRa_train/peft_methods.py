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
    DoRAConfig, AdaLoRAConfig, BitFitConfig, IA3Config, 
    PromptTuningConfig, AdapterConfig
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
        
        # Aplicar expertos
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        # Para cada experto seleccionado
        for i in range(2):  # Top-2
            expert_out = torch.zeros_like(out).view(-1, self.out_features)
            
            for expert_idx in range(self.config.num_experts):
                # Máscara para este experto
                mask = (router_indices[..., i] == expert_idx).view(-1)
                if mask.any():
                    expert_input = x_flat[mask]
                    
                    # Aplicar LoRA del experto
                    lora_out = self.lora_Bs[expert_idx](
                        self.lora_As[expert_idx](expert_input)
                    )
                    
                    # Ponderar por router
                    weight = router_weights[..., i].view(-1)[mask].unsqueeze(-1)
                    expert_out[mask] += lora_out * weight * self.config.lora_alpha
            
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

class AdaLoRALinear(BasePEFTModule):
    """AdaLoRA: LoRA con asignación adaptativa de rangos"""
    
    def __init__(self, in_features: int, out_features: int, config: 'AdaLoRAConfig'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Peso base
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Matrices LoRA con rango máximo inicial
        self.lora_E = nn.Parameter(torch.randn(config.init_r, in_features))
        self.lora_A = nn.Parameter(torch.randn(config.init_r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.init_r))
        
        # Máscaras para poda de rangos
        self.rank_mask = nn.Parameter(torch.ones(config.init_r), requires_grad=False)
        
        # Importancia de rangos
        self.rank_importance = nn.Parameter(torch.ones(config.init_r), requires_grad=False)
        
        # Inicialización
        nn.init.orthogonal_(self.lora_E)
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def update_rank_importance(self, step: int):
        """Actualiza importancia de rangos basado en gradientes"""
        if not self.training:
            return
        
        # Calcular importancia basada en magnitud de gradientes
        if self.lora_A.grad is not None and self.lora_B.grad is not None:
            importance = (self.lora_A.grad.norm(dim=1) + self.lora_B.grad.norm(dim=0)) / 2
            
            # Actualizar con momentum
            self.rank_importance.data = (
                self.config.beta1 * self.rank_importance.data +
                (1 - self.config.beta1) * importance
            )
    
    def prune_ranks(self, step: int):
        """Poda rangos menos importantes"""
        if step < self.config.tinit:
            return
        
        # Calcular número de rangos a mantener
        progress = min((step - self.config.tinit) / (self.config.tfinal - self.config.tinit), 1.0)
        current_r = int(self.config.init_r - progress * (self.config.init_r - self.config.target_r))
        
        # Seleccionar rangos más importantes
        _, indices = torch.topk(self.rank_importance, current_r)
        
        # Actualizar máscara
        self.rank_mask.zero_()
        self.rank_mask[indices] = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward base
        out = F.linear(x, self.weight)
        
        # Aplicar AdaLoRA con máscara
        masked_A = self.lora_A * self.rank_mask.unsqueeze(1)
        masked_B = self.lora_B * self.rank_mask.unsqueeze(0)
        
        # P @ A @ B donde P = E
        lora_out = F.linear(
            F.linear(x, masked_A),
            masked_B.T
        )
        
        # Regularización ortogonal en E
        if self.training:
            orth_loss = torch.norm(
                self.lora_E @ self.lora_E.T - torch.eye(self.config.init_r, device=x.device)
            )
            # Agregar a loss (manejado externamente)
            self.orth_loss = orth_loss * self.config.orth_reg_weight
        
        return out + lora_out * self.config.lora_alpha
    
    def merge_weights(self):
        with torch.no_grad():
            # Solo fusionar rangos activos
            active_indices = self.rank_mask.nonzero().squeeze()
            if len(active_indices) > 0:
                active_A = self.lora_A[active_indices]
                active_B = self.lora_B[:, active_indices]
                self.weight += active_B @ active_A * self.config.lora_alpha


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


# ============= Adapter =============

class Adapter(nn.Module):
    """Módulo adapter (bottleneck)"""
    
    def __init__(self, input_size: int, adapter_size: int, config: 'AdapterConfig'):
        super().__init__()
        self.input_size = input_size
        self.adapter_size = adapter_size
        self.config = config
        
        # Down-projection
        self.down_proj = nn.Linear(input_size, adapter_size)
        
        # Non-linearity
        if config.non_linearity == "relu":
            self.activation = nn.ReLU()
        elif config.non_linearity == "gelu":
            self.activation = nn.GELU()
        elif config.non_linearity == "swish":
            self.activation = nn.SiLU()
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_size, input_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.adapter_dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(input_size)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        if self.config.init_weights == "bert":
            # Inicialización estilo BERT
            nn.init.normal_(self.down_proj.weight, std=0.02)
            nn.init.normal_(self.up_proj.weight, std=0.02)
        elif self.config.init_weights == "xavier":
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
        
        # Bias a cero
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
        x = residual + x * self.config.scaling
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


class AdapterLayer(nn.Module):
    """Wrapper para insertar adapters en una capa"""
    
    def __init__(self, base_layer: nn.Module, adapter_size: int, 
                 config: 'AdapterConfig', adapter_type: str = "pfeiffer"):
        super().__init__()
        self.base_layer = base_layer
        self.adapter_type = adapter_type
        
        # Obtener tamaño de hidden
        if hasattr(base_layer, 'out_features'):
            hidden_size = base_layer.out_features
        elif hasattr(base_layer, 'normalized_shape'):
            hidden_size = base_layer.normalized_shape[0]
        else:
            raise ValueError("No se puede determinar hidden_size")
        
        # Crear adapter(s)
        if adapter_type == "pfeiffer":
            # Un adapter después de la capa
            self.adapter = Adapter(hidden_size, adapter_size, config)
        elif adapter_type == "houlsby":
            # Dos adapters: antes y después
            self.adapter_pre = Adapter(hidden_size, adapter_size, config)
            self.adapter_post = Adapter(hidden_size, adapter_size, config)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.adapter_type == "houlsby":
            x = self.adapter_pre(x)
        
        x = self.base_layer(x, *args, **kwargs)
        
        if self.adapter_type == "pfeiffer":
            x = self.adapter(x)
        elif self.adapter_type == "houlsby":
            x = self.adapter_post(x)
        
        return x
    
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
            
        elif config.method == PEFTMethod.ADALORA and isinstance(module, nn.Linear):
            new_module = AdaLoRALinear(module.in_features, module.out_features, config)
            new_module.weight.data = module.weight.data.clone()
            
        elif config.method == PEFTMethod.IA3 and isinstance(module, nn.Linear):
            is_feedforward = any(ffn in name for ffn in config.feedforward_modules or [])
            new_module = IA3Linear(module, config, is_feedforward)
            
        elif config.method == PEFTMethod.ADAPTER:
            if any(layer_type in name for layer_type in config.adapter_layers or []):
                new_module = AdapterLayer(module, config.adapter_size, config, config.adapter_type)
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