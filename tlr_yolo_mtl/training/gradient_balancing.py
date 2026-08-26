"""Multi-Task Gradient Balancing & Conflict Diagnostics (Ticket E46).

Provides:
1. GradNormBalancer (Chen et al., 2018):
   - Dynamically balances task loss weights w_i(t) to equalize task training rates
   - Computes relative inverse training rates r_i(t) = (L_i(t) / L_i(0)) / (mean_j L_j(t) / L_j(0))
   - Targets gradient norms G_W^{(i)}(t) -> avg_G * (r_i(t))^alpha

2. PCGradProjector (Yu et al., 2020):
   - Projects conflicting task gradients orthogonally:
     g_i <- g_i - (g_i . g_j / ||g_j||^2) * g_j  when  g_i . g_j < 0
   - Supports Layer-Restricted projection (e.g. Neck-Restricted to P2-P5 pyramid parameters)
     to reduce computational complexity from O(T^2 * |theta|) to shared representations.

3. LayerSpecificGradientDiagnostics:
   - Partitions network parameters into structural layers:
     * Shared Backbone (C2-C5)
     * Shared High-Res Neck (P2-P5)
     * Detection Heads (Detect convs)
     * Attribute Towers (State, Round, Maneuver ROIAlign)
     * Cross-Attention Relevance Head
   - Computes layer-stratified pairwise gradient cosine similarity matrices:
     C_{ij} = <g_i, g_j> / (||g_i|| * ||g_j||)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical task names for TLR-YOLO-MTL
TASK_NAMES: Tuple[str, ...] = (
    "Detection",
    "NWD",
    "State",
    "Round",
    "Maneuver",
    "Relevance",
)


def compute_gradient_cosine_similarity(
    g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-8
) -> float:
    """Compute cosine similarity between two flat 1D gradient vectors."""
    n1 = g1.norm(2)
    n2 = g2.norm(2)
    if n1 < eps or n2 < eps:
        return 0.0
    cos = torch.dot(g1, g2) / (n1 * n2)
    return float(cos.clamp(-1.0, 1.0).item())


def flatten_gradients(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> torch.Tensor:
    """Flatten a list of parameter gradients into a single 1D tensor."""
    chunks = []
    for g, p in zip(grads, params):
        if g is not None:
            chunks.append(g.reshape(-1))
        else:
            chunks.append(torch.zeros(p.numel(), device=device, dtype=p.dtype))
    if not chunks:
        return torch.zeros(0, device=device)
    return torch.cat(chunks)


def unflatten_gradients(
    flat_grad: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> List[torch.Tensor]:
    """Unflatten a 1D gradient vector back into a list of gradient tensors matching params."""
    result = []
    offset = 0
    for p in params:
        numel = p.numel()
        g = flat_grad[offset : offset + numel].reshape(p.shape)
        result.append(g)
        offset += numel
    return result


# =========================================================================
# 1. Parameter Layer Partitioning & Diagnostics
# =========================================================================

@dataclass
class LayerPartition:
    name: str
    params: List[torch.nn.Parameter]
    param_count: int = 0

    def __post_init__(self):
        self.param_count = sum(p.numel() for p in self.params)


def partition_model_parameters(model: torch.nn.Module) -> Dict[str, LayerPartition]:
    """Partition model parameters into semantic layer groups for multi-task gradient analysis."""
    backbone_params: List[torch.nn.Parameter] = []
    neck_params: List[torch.nn.Parameter] = []
    detect_head_params: List[torch.nn.Parameter] = []
    attribute_head_params: List[torch.nn.Parameter] = []
    relevance_head_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    # Check if model has standard ultralytics/wrapper structure
    inner_model = getattr(model, "model", model)
    layers = list(inner_model.children()) if hasattr(inner_model, "children") else [inner_model]

    # Look for UnifiedTrafficControlDetect module
    detect_mod = None
    for m in model.modules():
        if m.__class__.__name__ == "UnifiedTrafficControlDetect":
            detect_mod = m
            break

    if detect_mod is not None:
        # Detect module parameters
        detect_head_tensors = set()
        for p in detect_mod.cv2.parameters():
            if p.requires_grad:
                detect_head_params.append(p)
                detect_head_tensors.add(p)
        for p in detect_mod.cv3.parameters():
            if p.requires_grad:
                detect_head_params.append(p)
                detect_head_tensors.add(p)
        if hasattr(detect_mod, "dfl") and hasattr(detect_mod.dfl, "parameters"):
            for p in detect_mod.dfl.parameters():
                if p.requires_grad:
                    detect_head_params.append(p)
                    detect_head_tensors.add(p)

        attr_tensors = set()
        for name in ("state_heads", "round_heads", "maneuver_heads", "attribute_roi_align"):
            if hasattr(detect_mod, name):
                mod = getattr(detect_mod, name)
                for p in mod.parameters():
                    if p.requires_grad:
                        attribute_head_params.append(p)
                        attr_tensors.add(p)

        rel_tensors = set()
        for name in (
            "relevance_heads",
            "cross_attention",
            "adaptive_gate",
            "arrow_retrieval",
            "multiscale_fusion",
            "arrow_roi_align",
            "ego_lane_head",
        ):
            if hasattr(detect_mod, name):
                mod = getattr(detect_mod, name)
                for p in mod.parameters():
                    if p.requires_grad:
                        relevance_head_params.append(p)
                        rel_tensors.add(p)

        head_tensors = detect_head_tensors | attr_tensors | rel_tensors

        # Non-detect module parameters: partition into Backbone (0-10) and Neck (11-22)
        sub_layers = None
        if hasattr(model, "model") and isinstance(model.model, (nn.Sequential, list, nn.ModuleList)):
            sub_layers = list(model.model)
        elif isinstance(inner_model, (nn.Sequential, list, nn.ModuleList)):
            sub_layers = list(inner_model)
        elif hasattr(inner_model, "children"):
            sub_layers = list(inner_model.children())

        if sub_layers is not None and len(sub_layers) > 1:
            for idx, layer in enumerate(sub_layers):
                if layer is detect_mod or any(m is detect_mod for m in layer.modules()):
                    continue
                is_backbone = (idx <= 10)  # P2-P5 Backbone is typically layers 0..10
                for p in layer.parameters():
                    if p.requires_grad and p not in head_tensors:
                        if is_backbone:
                            backbone_params.append(p)
                        else:
                            neck_params.append(p)
        else:
            for p in model.parameters():
                if p.requires_grad and p not in head_tensors:
                    neck_params.append(p)
    else:
        # Fallback generic partitioning
        all_req = [p for p in model.parameters() if p.requires_grad]
        n = len(all_req)
        backbone_params = all_req[: n // 3]
        neck_params = all_req[n // 3 : 2 * (n // 3)]
        detect_head_params = all_req[2 * (n // 3) :]

    # Deduplicate lists preserving order
    def _dedup(plist: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
        seen = set()
        out = []
        for p in plist:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    return {
        "backbone": LayerPartition("Shared Backbone (C2-C5)", _dedup(backbone_params)),
        "neck": LayerPartition("Shared High-Res Neck (P2-P5)", _dedup(neck_params)),
        "shared_all": LayerPartition(
            "Shared Backbone + Neck", _dedup(backbone_params + neck_params)
        ),
        "detect_head": LayerPartition("Detection Heads", _dedup(detect_head_params)),
        "attribute_heads": LayerPartition("Attribute Heads (State/Round/Man)", _dedup(attribute_head_params)),
        "relevance_head": LayerPartition("Cross-Attention Relevance Head", _dedup(relevance_head_params)),
        "all_trainable": LayerPartition("Full Model (All Trainable)", [p for p in model.parameters() if p.requires_grad]),
    }


# =========================================================================
# 2. GradNorm Dynamic Loss Balancer (Chen et al., 2018)
# =========================================================================

class GradNormBalancer:
    """Dynamic Multi-Task Loss Weight Balancer via Gradient Normalization (GradNorm).

    Equalizes the pace at which different task losses train by penalizing tasks whose
    gradient norms deviate from their target training rates.

    Target loss weights w_i are updated dynamically such that:
      w_i(t) * L_i(t) balance the gradients at shared representation W.
    """

    def __init__(
        self,
        task_names: Sequence[str] = TASK_NAMES,
        initial_weights: Optional[Mapping[str, float]] = None,
        alpha: float = 1.5,
        update_rate: float = 0.025,
        target_layer_params: Optional[Sequence[torch.nn.Parameter]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.task_names = tuple(task_names)
        self.num_tasks = len(self.task_names)
        self.alpha = float(alpha)
        self.update_rate = float(update_rate)
        self.device = device

        if initial_weights is not None:
            init_w = [float(initial_weights.get(name, 1.0)) for name in self.task_names]
        else:
            init_w = [1.0] * self.num_tasks

        # Normalize initial weights to sum to num_tasks
        scale = self.num_tasks / max(sum(init_w), 1e-6)
        self.weights = np.array([w * scale for w in init_w], dtype=np.float32)

        self.initial_losses: Optional[np.ndarray] = None
        self.current_losses: np.ndarray = np.zeros(self.num_tasks, dtype=np.float32)
        self.history_weights: List[Dict[str, float]] = []
        self.history_losses: List[Dict[str, float]] = []
        self.step_count = 0

    def set_initial_losses(self, losses: Sequence[float]) -> None:
        """Lock reference initial losses L_i(0) at step 0."""
        self.initial_losses = np.maximum(np.array(losses, dtype=np.float32), 1e-6)

    def get_weights_dict(self) -> Dict[str, float]:
        """Return current task weights as a dictionary."""
        return {name: float(self.weights[i]) for i, name in enumerate(self.task_names)}

    def update_weights(
        self,
        task_losses: Sequence[float],
        task_grad_norms: Sequence[float],
    ) -> Dict[str, float]:
        """Perform one GradNorm update step given current loss values and gradient norms."""
        loss_arr = np.maximum(np.array(task_losses, dtype=np.float32), 1e-6)
        norm_arr = np.maximum(np.array(task_grad_norms, dtype=np.float32), 1e-6)
        self.current_losses = loss_arr

        if self.initial_losses is None:
            self.set_initial_losses(loss_arr)

        assert self.initial_losses is not None

        # 1. Loss ratios L_i(t) / L_i(0)
        loss_ratios = loss_arr / self.initial_losses
        mean_loss_ratio = float(np.mean(loss_ratios))

        # 2. Relative inverse training rates r_i(t)
        rel_train_rates = loss_ratios / max(mean_loss_ratio, 1e-6)

        # 3. Target gradient norms G_target = mean_G * (r_i(t))^alpha
        mean_grad_norm = float(np.mean(norm_arr))
        target_grad_norms = mean_grad_norm * (rel_train_rates ** self.alpha)

        # 4. Gradient of L_grad = sum_i |G_i - G_target_i| w.r.t w_i
        # d L_grad / d w_i ~ (norm_arr_i - target_i)
        grad_w = (norm_arr - target_grad_norms) / max(mean_grad_norm, 1e-6)

        # Update weights using gradient descent step
        self.weights = self.weights - self.update_rate * grad_w

        # Clip and renormalize to maintain sum(w_i) = num_tasks
        self.weights = np.maximum(self.weights, 0.05)
        self.weights = self.weights * (self.num_tasks / np.sum(self.weights))

        self.step_count += 1
        w_dict = self.get_weights_dict()
        self.history_weights.append(w_dict)
        self.history_losses.append({name: float(loss_arr[i]) for i, name in enumerate(self.task_names)})
        return w_dict


# =========================================================================
# 3. PCGrad: Projected Conflicting Gradients (Yu et al., 2020)
# =========================================================================

class PCGradProjector:
    """Projected Conflicting Gradients (PCGrad) with Layer-Restricted Execution.

    When gradients between task i and task j conflict (<g_i, g_j> < 0),
    projects g_i onto the normal plane of g_j:
      g_i <- g_i - (<g_i, g_j> / ||g_j||^2) * g_j

    In Neck-Restricted mode, projection is performed exclusively on the shared
    multi-scale pyramid neck (P2-P5), avoiding O(T^2 * |theta|) overhead on deep backbone
    or task-specific private heads.
    """

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        device: torch.device = torch.device("cpu"),
    ):
        self.params = [p for p in params if p.requires_grad]
        self.num_params = sum(p.numel() for p in self.params)
        self.device = device
        self.total_conflict_count = 0
        self.total_comparisons = 0

    def project_conflicting_gradients(
        self,
        task_gradients: Sequence[torch.Tensor],
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Project a list of flat 1D gradient vectors from multiple tasks.

        Args:
            task_gradients: List of 1D tensors, each of shape (num_params,)
            shuffle: Whether to randomize task evaluation order (as in Yu et al., 2020)

        Returns:
            fused_gradient: 1D tensor representing the sum of projected task gradients.
            telemetry: Dict with conflict counts, cosine alignments, and projection stats.
        """
        num_tasks = len(task_gradients)
        if num_tasks <= 1:
            fused = task_gradients[0] if num_tasks == 1 else torch.zeros(self.num_params, device=self.device)
            return fused, {"conflicts_detected": 0, "total_pairs": 0, "conflict_rate": 0.0}

        # Clone gradients
        projected = [g.clone() for g in task_gradients]
        task_indices = list(range(num_tasks))

        step_conflicts = 0
        step_comparisons = 0

        order = list(range(num_tasks))
        if shuffle:
            random.shuffle(order)

        for i in order:
            g_i = projected[i]
            other_tasks = [j for j in range(num_tasks) if j != i]
            if shuffle:
                random.shuffle(other_tasks)

            for j in other_tasks:
                g_j = task_gradients[j]  # Original or current g_j
                step_comparisons += 1

                dot = torch.dot(g_i, g_j)
                if dot < 0.0:
                    # Conflict detected: project g_i orthogonal to g_j
                    step_conflicts += 1
                    norm_sq = torch.dot(g_j, g_j).clamp_min(1e-12)
                    g_i = g_i - (dot / norm_sq) * g_j
                    projected[i] = g_i

        self.total_conflict_count += step_conflicts
        self.total_comparisons += step_comparisons

        fused_gradient = torch.stack(projected, dim=0).sum(dim=0)
        conflict_rate = float(step_conflicts / max(step_comparisons, 1))

        telemetry = {
            "conflicts_detected": step_conflicts,
            "total_pairs": step_comparisons,
            "conflict_rate": conflict_rate,
            "cumulative_conflict_rate": float(
                self.total_conflict_count / max(self.total_comparisons, 1)
            ),
        }
        return fused_gradient, telemetry


# =========================================================================
# 4. Multi-Task Gradient Diagnostics & Synergy Matrix Evaluator
# =========================================================================

class MultiTaskGradientDiagnostics:
    """Evaluates gradient cosine similarities and norm distributions across layers and epochs."""

    def __init__(
        self,
        model: torch.nn.Module,
        task_names: Sequence[str] = TASK_NAMES,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.task_names = tuple(task_names)
        self.device = device
        self.partitions = partition_model_parameters(model)

    def compute_task_layer_gradients(
        self,
        task_losses: Sequence[torch.Tensor],
        partition_key: str = "shared_all",
    ) -> List[torch.Tensor]:
        """Compute flat gradient vectors for each task with respect to a parameter partition."""
        partition = self.partitions.get(partition_key, self.partitions["shared_all"])
        params = partition.params

        if not params:
            raise ValueError(f"Partition {partition_key} has 0 trainable parameters.")

        task_grads: List[torch.Tensor] = []
        num_tasks = len(task_losses)

        for idx, loss in enumerate(task_losses):
            is_last = (idx == num_tasks - 1)
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=(not is_last),
                allow_unused=True,
            )
            flat = flatten_gradients(grads, params, self.device)
            task_grads.append(flat)

        return task_grads

    def compute_all_partition_gradients(
        self,
        task_losses: Sequence[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """Compute task gradients across all layer partitions in a single backward pass per task."""
        all_params = self.partitions["all_trainable"].params
        num_tasks = len(task_losses)
        param_to_idx = {id(p): idx for idx, p in enumerate(all_params)}

        partition_indices = {
            k: [param_to_idx[id(p)] for p in part.params if id(p) in param_to_idx]
            for k, part in self.partitions.items()
        }

        partition_grads: Dict[str, List[torch.Tensor]] = {k: [] for k in self.partitions}

        for task_idx, loss in enumerate(task_losses):
            is_last = (task_idx == num_tasks - 1)
            grads = torch.autograd.grad(
                loss,
                all_params,
                retain_graph=(not is_last),
                allow_unused=True,
            )
            for k, p_indices in partition_indices.items():
                p_grads = [grads[i] for i in p_indices]
                flat = flatten_gradients(p_grads, self.partitions[k].params, self.device)
                partition_grads[k].append(flat)

        return partition_grads

    def compute_pairwise_cosine_matrix(
        self, task_grads: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute the N x N pairwise cosine similarity matrix from task gradient vectors."""
        n = len(task_grads)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = compute_gradient_cosine_similarity(
                        task_grads[i], task_grads[j]
                    )
        return matrix
