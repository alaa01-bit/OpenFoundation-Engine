"""
OpenFoundation-Engine: Distributed HPC Training Dynamics & Diagnostics
Implements algorithms for large-scale foundation model pretraining:
- Cosine Annealing Learning Rate Scheduler with Linear Warmup
- Gradient Norm Monitoring & Loss Spike Anomaly Detection
- Distributed VRAM Memory Footprint Estimator (ZeRO-1, ZeRO-2, ZeRO-3 / FSDP, Tensor Parallelism)
- Hardware Model FLOPs Utilization (MFU) Calculator for HPC Clusters (A100 / H100)
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class SchedulerConfig:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 100000


class CosineWarmupScheduler:
    """Cosine learning rate decay with linear warmup phase."""
    def __init__(self, config: SchedulerConfig):
        self.cfg = config

    def get_lr(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.max_lr * float(step) / max(1, self.cfg.warmup_steps)
        if step > self.cfg.max_steps:
            return self.cfg.min_lr

        decay_ratio = float(step - self.cfg.warmup_steps) / float(self.cfg.max_steps - self.cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.cfg.min_lr + coeff * (self.cfg.max_lr - self.cfg.min_lr)


class LossSpikeMonitor:
    """
    Monitors training dynamics in real-time.
    Detects sudden gradient explosions, loss spikes, and instability during pretraining.
    """
    def __init__(self, window_size: int = 20, spike_z_threshold: float = 3.0):
        self.window_size = window_size
        self.spike_z_threshold = spike_z_threshold
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []

    def record_step(self, step: int, loss: float, grad_norm: float) -> Dict[str, Any]:
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

        alert = "NORMAL"
        z_score = 0.0

        if len(self.loss_history) > self.window_size:
            recent_losses = self.loss_history[-self.window_size:-1]
            mean_loss = float(np.mean(recent_losses))
            std_loss = float(np.std(recent_losses)) + 1e-8
            z_score = (loss - mean_loss) / std_loss

            if z_score > self.spike_z_threshold or math.isnan(loss):
                alert = "LOSS_SPIKE_DETECTED"
            elif grad_norm > 5.0:
                alert = "GRADIENT_EXPLOSION_WARNING"

        return {
            "step": step,
            "loss": round(loss, 4),
            "grad_norm": round(grad_norm, 4),
            "z_score": round(z_score, 2),
            "status": alert,
            "recommended_action": "REVERT_TO_LAST_STABLE_CHECKPOINT" if alert == "LOSS_SPIKE_DETECTED" else "CONTINUE"
        }


class DistributedMemoryPlanner:
    """
    Calculates GPU VRAM memory requirements across standard distributed strategies:
    - Standard DDP (Data Parallel)
    - DeepSpeed ZeRO-1 (Partition Optimizer States)
    - DeepSpeed ZeRO-2 (Partition Optimizer + Gradients)
    - DeepSpeed ZeRO-3 / PyTorch FSDP (Partition Optimizer + Gradients + Model Weights)
    """
    @staticmethod
    def calculate_memory_footprint(
        param_count: int,
        world_size: int = 8,
        bytes_per_param: int = 2,  # BF16 / FP16
        batch_size: int = 4,
        seq_len: int = 4096,
        d_model: int = 4096,
        n_layers: int = 32
    ) -> Dict[str, Any]:
        """
        Estimates per-GPU memory breakdown in Gigabytes (GB).
        Weights: 2 bytes per param
        Gradients: 2 bytes per param
        AdamW Optimizer states: 12 bytes per param (FP32 master weights (4B) + momentum (4B) + variance (4B))
        Activations: standard rough heuristic ~ (seq_len * batch_size * d_model * layers * overhead)
        """
        bytes_to_gb = 1024 ** 3

        weights_bytes = param_count * bytes_per_param
        grad_bytes = param_count * bytes_per_param
        adam_bytes = param_count * 12

        # Activation memory estimate per GPU with activation checkpointing
        activation_gb = (batch_size * seq_len * d_model * n_layers * 10) / bytes_to_gb

        # 1. Standard DDP (Full replication on each GPU)
        ddp_static_gb = (weights_bytes + grad_bytes + adam_bytes) / bytes_to_gb

        # 2. ZeRO-1 (Optimizer partitioned across world_size)
        zero1_static_gb = (weights_bytes + grad_bytes + (adam_bytes / world_size)) / bytes_to_gb

        # 3. ZeRO-2 (Optimizer + Gradients partitioned)
        zero2_static_gb = (weights_bytes + ((grad_bytes + adam_bytes) / world_size)) / bytes_to_gb

        # 4. ZeRO-3 / FSDP (Optimizer + Gradients + Parameters partitioned)
        zero3_static_gb = (weights_bytes + grad_bytes + adam_bytes) / (world_size * bytes_to_gb)

        return {
            "parameter_count": param_count,
            "world_size_gpus": world_size,
            "ddp_total_vram_gb": round(ddp_static_gb + activation_gb, 2),
            "zero1_vram_gb": round(zero1_static_gb + activation_gb, 2),
            "zero2_vram_gb": round(zero2_static_gb + activation_gb, 2),
            "fsdp_zero3_vram_gb": round(zero3_static_gb + activation_gb, 2),
            "activation_vram_gb": round(activation_gb, 2),
            "static_model_vram_gb": round(weights_bytes / bytes_to_gb, 2)
        }

    @staticmethod
    def calculate_mfu(
        tokens_per_second: float,
        param_count: int,
        peak_tflops_per_gpu: float = 989.0,  # NVIDIA H100 SXM5 BF16 peak
        num_gpus: int = 8
    ) -> Dict[str, float]:
        """
        Calculates Model FLOPs Utilization (MFU) (Chowdhery et al., 2022).
        MFU = (6 * tokens_per_sec * param_count) / (num_gpus * peak_flops_per_sec)
        A pretraining cluster typically achieves 35% - 55% MFU.
        """
        flops_achieved = 6.0 * tokens_per_second * param_count
        theoretical_peak = num_gpus * peak_tflops_per_gpu * 1e12

        mfu_percent = (flops_achieved / theoretical_peak) * 100.0

        return {
            "tokens_per_second": round(tokens_per_second, 1),
            "flops_achieved_tflops": round(flops_achieved / 1e12, 2),
            "theoretical_peak_tflops": round(theoretical_peak / 1e12, 2),
            "mfu_percent": round(mfu_percent, 2)
        }

