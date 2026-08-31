"""Post-training scalar and multi-task temperature scaling & conformal risk control for frozen model logits."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    temperature: float
    loss_before: float
    loss_after: float
    valid_samples: int
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None
    mce_before: float | None = None
    mce_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemperatureFit:
        return cls(
            temperature=float(data["temperature"]),
            loss_before=float(data["loss_before"]),
            loss_after=float(data["loss_after"]),
            valid_samples=int(data["valid_samples"]),
            ece_before=float(data["ece_before"]) if data.get("ece_before") is not None else None,
            ece_after=float(data["ece_after"]) if data.get("ece_after") is not None else None,
            brier_before=float(data["brier_before"]) if data.get("brier_before") is not None else None,
            brier_after=float(data["brier_after"]) if data.get("brier_after") is not None else None,
            mce_before=float(data["mce_before"]) if data.get("mce_before") is not None else None,
            mce_after=float(data["mce_after"]) if data.get("mce_after") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ConformalThresholdResult:
    target_recall: float
    alpha_risk: float
    fitted_threshold: float
    empirical_risk: float
    calibration_recall: float
    calibration_precision: float
    finite_sample_bound: float
    valid_samples: int
    guarantee_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalThresholdResult:
        return cls(
            target_recall=float(data["target_recall"]),
            alpha_risk=float(data["alpha_risk"]),
            fitted_threshold=float(data["fitted_threshold"]),
            empirical_risk=float(data["empirical_risk"]),
            calibration_recall=float(data["calibration_recall"]),
            calibration_precision=float(data["calibration_precision"]),
            finite_sample_bound=float(data["finite_sample_bound"]),
            valid_samples=int(data["valid_samples"]),
            guarantee_satisfied=bool(data["guarantee_satisfied"]),
        )


# =========================================================================
# Metric Computation Functions
# =========================================================================

def _to_numpy(x: Sequence | np.ndarray | torch.Tensor, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def compute_nll(
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence | np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute Negative Log-Likelihood for binary or multi-class predictions."""
    y = _to_numpy(targets, dtype=np.int64).reshape(-1)
    p = _to_numpy(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = np.clip(p.reshape(-1), eps, 1.0 - eps)
        loss = -(y * np.log(p) + (1 - y) * np.log(1.0 - p))
        return float(np.mean(loss)) if loss.size else float("nan")
    elif p.ndim == 2:
        if p.shape[0] != y.shape[0]:
            raise ValueError(f"targets shape {y.shape} does not match probs shape {p.shape}")
        p = np.clip(p, eps, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        row_indices = np.arange(len(y))
        valid_mask = (y >= 0) & (y < p.shape[1])
        if not np.any(valid_mask):
            return float("nan")
        picked_probs = p[row_indices[valid_mask], y[valid_mask]]
        return float(-np.mean(np.log(picked_probs)))
    else:
        raise ValueError(f"probabilities must be 1D or 2D, got shape {p.shape}")


def compute_multiclass_ece(
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence | np.ndarray | torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) for multi-class predictions using top confidence."""
    y = _to_numpy(targets, dtype=np.int64).reshape(-1)
    p = _to_numpy(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"targets shape {y.shape} does not match probs shape {p.shape}")
    
    valid_mask = (y >= 0) & (y < p.shape[1])
    if not np.any(valid_mask):
        return 0.0
    y = y[valid_mask]
    p = p[valid_mask]

    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    accuracies = (predictions == y).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(confidences, edges[1:-1]), bins - 1)
    error = 0.0
    total = float(len(y))
    for index in range(bins):
        selected = bucket == index
        count = int(np.sum(selected))
        if count == 0:
            continue
        bin_acc = float(np.mean(accuracies[selected]))
        bin_conf = float(np.mean(confidences[selected]))
        error += (count / total) * abs(bin_acc - bin_conf)
    return float(error)


def compute_classwise_ece(
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence | np.ndarray | torch.Tensor,
    *,
    num_classes: int = 4,
    bins: int = 15,
) -> dict[int, float]:
    """Class-wise Expected Calibration Error (one-vs-rest per class)."""
    y = _to_numpy(targets, dtype=np.int64).reshape(-1)
    p = _to_numpy(probabilities, dtype=np.float64)
    if p.ndim == 1 and num_classes == 2:
        p = np.stack([1.0 - p, p], axis=1)
    
    valid_mask = (y >= 0) & (y < num_classes)
    y = y[valid_mask]
    p = p[valid_mask]
    
    class_eces: dict[int, float] = {}
    edges = np.linspace(0.0, 1.0, bins + 1)
    for c in range(num_classes):
        target_c = (y == c).astype(float)
        prob_c = p[:, c]
        bucket = np.minimum(np.digitize(prob_c, edges[1:-1]), bins - 1)
        err = 0.0
        total = float(len(y))
        for b in range(bins):
            sel = bucket == b
            cnt = int(np.sum(sel))
            if cnt == 0:
                continue
            acc_b = float(np.mean(target_c[sel]))
            conf_b = float(np.mean(prob_c[sel]))
            err += (cnt / total) * abs(acc_b - conf_b)
        class_eces[c] = float(err)
    return class_eces


def compute_maximum_calibration_error(
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence | np.ndarray | torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    """Maximum Calibration Error (MCE) across all confidence bins."""
    y = _to_numpy(targets, dtype=np.int64).reshape(-1)
    p = _to_numpy(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    
    valid_mask = (y >= 0) & (y < p.shape[1])
    if not np.any(valid_mask):
        return 0.0
    y = y[valid_mask]
    p = p[valid_mask]

    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    accuracies = (predictions == y).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(confidences, edges[1:-1]), bins - 1)
    max_err = 0.0
    for index in range(bins):
        selected = bucket == index
        if not np.any(selected):
            continue
        bin_acc = float(np.mean(accuracies[selected]))
        bin_conf = float(np.mean(confidences[selected]))
        err = abs(bin_acc - bin_conf)
        if err > max_err:
            max_err = err
    return float(max_err)


def compute_brier_multiclass(
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence | np.ndarray | torch.Tensor,
) -> float:
    """Multi-class Brier score: mean squared error over one-hot vectors."""
    y = _to_numpy(targets, dtype=np.int64).reshape(-1)
    p = _to_numpy(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    
    valid_mask = (y >= 0) & (y < p.shape[1])
    if not np.any(valid_mask):
        return float("nan")
    y = y[valid_mask]
    p = p[valid_mask]

    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    sample_brier = np.sum((p - one_hot) ** 2, axis=1)
    return float(np.mean(sample_brier))


# =========================================================================
# Core Temperature Scaling
# =========================================================================

def _valid(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets = targets.reshape(-1).long()
    if logits.ndim == 1:
        logits = logits.reshape(-1)
    elif logits.ndim == 2:
        if logits.shape[0] != targets.numel():
            raise ValueError("logit and target sample counts differ")
    else:
        raise ValueError("calibration logits must have shape [N] or [N, C]")
    mask = targets.ge(0)
    return logits[mask].float(), targets[mask]


def _nll(logits: torch.Tensor, targets: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    scaled = logits / temperature
    if logits.ndim == 1:
        return F.binary_cross_entropy_with_logits(scaled, targets.float())
    return F.cross_entropy(scaled, targets)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    grid_points: int = 500,
    compute_diagnostics: bool = True,
) -> TemperatureFit:
    """Fit one positive scalar temperature by deterministic log-space search."""
    values, labels = _valid(logits, targets)
    if not labels.numel():
        raise ValueError("temperature fitting requires at least one valid target")
    candidates = torch.logspace(
        torch.log10(torch.tensor(minimum)),
        torch.log10(torch.tensor(maximum)),
        grid_points,
        device=values.device,
    )
    losses = torch.stack([_nll(values, labels, value) for value in candidates])
    best = int(losses.argmin())
    before = _nll(values, labels, torch.tensor(1.0, device=values.device))
    T_star = float(candidates[best])

    ece_before, ece_after = None, None
    brier_before, brier_after = None, None
    mce_before, mce_after = None, None

    if compute_diagnostics:
        labels_np = labels.cpu().numpy()
        if values.ndim == 1:
            raw_p = torch.sigmoid(values).cpu().numpy()
            cal_p = torch.sigmoid(values / T_star).cpu().numpy()
            raw_p2 = np.stack([1.0 - raw_p, raw_p], axis=1)
            cal_p2 = np.stack([1.0 - cal_p, cal_p], axis=1)
        else:
            raw_p2 = F.softmax(values, dim=-1).cpu().numpy()
            cal_p2 = F.softmax(values / T_star, dim=-1).cpu().numpy()
        
        ece_before = compute_multiclass_ece(labels_np, raw_p2)
        ece_after = compute_multiclass_ece(labels_np, cal_p2)
        brier_before = compute_brier_multiclass(labels_np, raw_p2)
        brier_after = compute_brier_multiclass(labels_np, cal_p2)
        mce_before = compute_maximum_calibration_error(labels_np, raw_p2)
        mce_after = compute_maximum_calibration_error(labels_np, cal_p2)

    return TemperatureFit(
        temperature=T_star,
        loss_before=float(before),
        loss_after=float(losses[best]),
        valid_samples=int(labels.numel()),
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=brier_before,
        brier_after=brier_after,
        mce_before=mce_before,
        mce_after=mce_after,
    )


def apply_temperature(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Scale logits by temperature value(s)."""
    if isinstance(temperature, (int, float)):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
    elif isinstance(temperature, torch.Tensor):
        if (temperature <= 0).any():
            raise ValueError("temperature tensor must contain positive values")
    else:
        raise TypeError(f"unsupported temperature type: {type(temperature)}")
    return logits / temperature


# =========================================================================
# Multi-Task Temperature Calibrator
# =========================================================================

class MultiTaskTemperatureCalibrator:
    """Manages post-hoc temperature scaling for Traffic Light State and Relevance heads."""

    def __init__(
        self,
        state_temperature: float = 1.0,
        relevance_temperature: float = 1.0,
        state_temperature_vector: Sequence[float] | None = None,
    ) -> None:
        self.state_temperature = float(state_temperature)
        self.relevance_temperature = float(relevance_temperature)
        self.state_temperature_vector = (
            [float(v) for v in state_temperature_vector] if state_temperature_vector is not None else None
        )
        self.fit_state: TemperatureFit | None = None
        self.fit_relevance: TemperatureFit | None = None

    def fit_state_head(
        self,
        state_logits: torch.Tensor,
        state_targets: torch.Tensor,
        *,
        minimum: float = 0.05,
        maximum: float = 20.0,
        grid_points: int = 500,
    ) -> TemperatureFit:
        fit = fit_temperature(state_logits, state_targets, minimum=minimum, maximum=maximum, grid_points=grid_points)
        self.state_temperature = fit.temperature
        self.fit_state = fit
        return fit

    def fit_relevance_head(
        self,
        relevance_logits: torch.Tensor,
        relevance_targets: torch.Tensor,
        *,
        minimum: float = 0.05,
        maximum: float = 20.0,
        grid_points: int = 500,
    ) -> TemperatureFit:
        fit = fit_temperature(relevance_logits, relevance_targets, minimum=minimum, maximum=maximum, grid_points=grid_points)
        self.relevance_temperature = fit.temperature
        self.fit_relevance = fit
        return fit

    def fit(
        self,
        state_logits: torch.Tensor | None = None,
        state_targets: torch.Tensor | None = None,
        relevance_logits: torch.Tensor | None = None,
        relevance_targets: torch.Tensor | None = None,
    ) -> dict[str, TemperatureFit]:
        results: dict[str, TemperatureFit] = {}
        if state_logits is not None and state_targets is not None:
            results["state"] = self.fit_state_head(state_logits, state_targets)
        if relevance_logits is not None and relevance_targets is not None:
            results["relevance"] = self.fit_relevance_head(relevance_logits, relevance_targets)
        return results

    def calibrate_state_logits(self, state_logits: torch.Tensor) -> torch.Tensor:
        if self.state_temperature_vector is not None:
            t = state_logits.new_tensor(self.state_temperature_vector).view(1, -1, 1)
            return state_logits / t
        return state_logits / self.state_temperature

    def calibrate_state_probabilities(self, state_logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return F.softmax(self.calibrate_state_logits(state_logits), dim=dim)

    def calibrate_relevance_logits(self, relevance_logits: torch.Tensor) -> torch.Tensor:
        return relevance_logits / self.relevance_temperature

    def calibrate_relevance_probabilities(self, relevance_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.calibrate_relevance_logits(relevance_logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_temperature": self.state_temperature,
            "relevance_temperature": self.relevance_temperature,
            "state_temperature_vector": self.state_temperature_vector,
            "fit_state": self.fit_state.to_dict() if self.fit_state is not None else None,
            "fit_relevance": self.fit_relevance.to_dict() if self.fit_relevance is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiTaskTemperatureCalibrator:
        calibrator = cls(
            state_temperature=data.get("state_temperature", 1.0),
            relevance_temperature=data.get("relevance_temperature", 1.0),
            state_temperature_vector=data.get("state_temperature_vector"),
        )
        if data.get("fit_state") is not None:
            calibrator.fit_state = TemperatureFit.from_dict(data["fit_state"])
        if data.get("fit_relevance") is not None:
            calibrator.fit_relevance = TemperatureFit.from_dict(data["fit_relevance"])
        return calibrator

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> MultiTaskTemperatureCalibrator:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =========================================================================
# Split Conformal Prediction for Multi-Class State
# =========================================================================

class ConformalStatePredictor:
    """Split Conformal Prediction for Traffic Light State (Red/Yellow/Green/Off).

    Produces valid prediction sets C_alpha(x) guaranteeing marginal coverage:
        P(Y in C_alpha(X)) >= 1 - alpha
    under the exchangeability assumption, with exact finite-sample quantile correction:
        q_hat = Quantile({s_i}, ceil((n + 1) * (1 - alpha)) / n)
    """

    def __init__(self, method: str = "lac") -> None:
        if method not in ("lac", "aps"):
            raise ValueError(f"unsupported conformal method: {method} (choose 'lac' or 'aps')")
        self.method = method
        self.nonconformity_scores: np.ndarray | None = None
        self.quantiles: dict[float, float] = {}

    def fit(
        self,
        probabilities: Sequence | np.ndarray | torch.Tensor,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> None:
        y = _to_numpy(targets, dtype=np.int64).reshape(-1)
        p = _to_numpy(probabilities, dtype=np.float64)
        if p.ndim == 1:
            p = np.stack([1.0 - p, p], axis=1)
        
        valid_mask = (y >= 0) & (y < p.shape[1])
        if not np.any(valid_mask):
            raise ValueError("ConformalStatePredictor requires at least one valid calibration sample")
        y = y[valid_mask]
        p = p[valid_mask]
        n = len(y)

        if self.method == "lac":
            # Least Ambiguous Set-Valued Classifier: s_i = 1 - p(y_i)
            row_idx = np.arange(n)
            true_probs = p[row_idx, y]
            scores = 1.0 - true_probs
        elif self.method == "aps":
            # Adaptive Prediction Sets: cumulative sum of sorted probabilities up to true class
            scores = np.zeros(n, dtype=np.float64)
            for i in range(n):
                sort_idx = np.argsort(-p[i])
                cumsum = np.cumsum(p[i, sort_idx])
                rank = int(np.where(sort_idx == y[i])[0][0])
                scores[i] = cumsum[rank]
        else:
            raise ValueError(f"unknown method: {self.method}")

        self.nonconformity_scores = np.sort(scores)
        self.quantiles.clear()

    def get_quantile(self, alpha: float) -> float:
        if self.nonconformity_scores is None or len(self.nonconformity_scores) == 0:
            raise RuntimeError("ConformalStatePredictor must be fitted before computing quantiles")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        
        if alpha in self.quantiles:
            return self.quantiles[alpha]

        n = len(self.nonconformity_scores)
        # Finite-sample exact quantile index: ceil((n + 1) * (1 - alpha)) / n
        level = math.ceil((n + 1) * (1.0 - alpha)) / n
        level = min(max(level, 0.0), 1.0)
        # Quantile index in sorted array (0-indexed)
        idx = min(int(math.ceil(level * n)) - 1, n - 1)
        idx = max(0, idx)
        q = float(self.nonconformity_scores[idx])
        self.quantiles[alpha] = q
        return q

    def predict_set(
        self,
        probabilities: Sequence | np.ndarray | torch.Tensor,
        alpha: float = 0.05,
    ) -> np.ndarray | torch.Tensor:
        """Return boolean mask of shape [N, num_classes] indicating included classes in C_alpha."""
        q = self.get_quantile(alpha)
        is_torch = isinstance(probabilities, torch.Tensor)
        p = _to_numpy(probabilities, dtype=np.float64)
        if p.ndim == 1:
            p = np.stack([1.0 - p, p], axis=1)

        if self.method == "lac":
            threshold = max(0.0, 1.0 - q)
            prediction_sets = p >= threshold
            empty_mask = ~np.any(prediction_sets, axis=-1)
            if np.any(empty_mask):
                max_indices = np.argmax(p, axis=-1)
                for row_idx in np.where(empty_mask)[0]:
                    prediction_sets[row_idx, max_indices[row_idx]] = True
        elif self.method == "aps":
            prediction_sets = np.zeros_like(p, dtype=bool)
            for i in range(len(p)):
                sort_idx = np.argsort(-p[i])
                cumsum = np.cumsum(p[i, sort_idx])
                included_sorted = np.where(cumsum <= q)[0]
                k = len(included_sorted) + 1
                k = min(k, p.shape[1])
                prediction_sets[i, sort_idx[:k]] = True

        if is_torch:
            return torch.from_numpy(prediction_sets)
        return prediction_sets

    def evaluate_prediction_sets(
        self,
        probabilities: Sequence | np.ndarray | torch.Tensor,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """Compute empirical coverage, average set size, and singleton/empty rates on test split."""
        y = _to_numpy(targets, dtype=np.int64).reshape(-1)
        p = _to_numpy(probabilities, dtype=np.float64)
        if p.ndim == 1:
            p = np.stack([1.0 - p, p], axis=1)
        
        valid_mask = (y >= 0) & (y < p.shape[1])
        y = y[valid_mask]
        p = p[valid_mask]

        sets = self.predict_set(p, alpha=alpha)
        if isinstance(sets, torch.Tensor):
            sets = sets.cpu().numpy()
        
        n = len(y)
        if n == 0:
            return {"coverage": 0.0, "avg_set_size": 0.0, "singleton_rate": 0.0, "empty_rate": 0.0}

        covered = sets[np.arange(n), y]
        set_sizes = np.sum(sets, axis=1)

        coverage = float(np.mean(covered))
        avg_set_size = float(np.mean(set_sizes))
        singleton_rate = float(np.mean(set_sizes == 1))
        empty_rate = float(np.mean(set_sizes == 0))

        return {
            "target_coverage": float(1.0 - alpha),
            "empirical_coverage": coverage,
            "coverage_gap": coverage - (1.0 - alpha),
            "avg_set_size": avg_set_size,
            "singleton_rate": singleton_rate,
            "empty_rate": empty_rate,
        }


# =========================================================================
# Conformal Risk Control for Relevant Red Recall
# =========================================================================

class ConformalRiskController:
    """Conformal Risk Control (CRC) & Learn-then-Test Safety Solver.

    Guarantees that the expected False Negative Rate (FNR = 1 - Recall) on safety-critical
    traffic light states (Relevant Red) satisfies:
        E[L(tau_hat)] <= alpha_risk
    under exchangeability with finite-sample inflation:
        (n / (n + 1)) * R_hat(tau) + 1 / (n + 1) <= alpha_risk
    yielding mathematically certified Relevant Red Recall >= 1 - alpha_risk.
    """

    @staticmethod
    def solve_safety_threshold(
        scores: Sequence[float] | np.ndarray | torch.Tensor,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        target_recall: float = 0.975,
        alpha_risk: float | None = None,
        min_threshold: float = 0.001,
        max_threshold: float = 0.999,
        grid_resolution: int = 1000,
    ) -> ConformalThresholdResult:
        """Find the maximum precision threshold tau_hat guaranteeing Expected Recall >= target_recall."""
        if alpha_risk is None:
            alpha_risk = 1.0 - target_recall
        
        y = _to_numpy(targets, dtype=np.int64).reshape(-1)
        s = _to_numpy(scores, dtype=np.float64).reshape(-1)

        if len(y) != len(s):
            raise ValueError(f"targets length {len(y)} does not match scores length {len(s)}")

        pos_mask = y == 1
        n_pos = int(np.sum(pos_mask))
        if n_pos == 0:
            raise ValueError("Conformal Risk Control requires at least one positive safety sample")

        pos_scores = s[pos_mask]

        candidates = np.linspace(min_threshold, max_threshold, grid_resolution)
        best_tau = float(min_threshold)
        best_risk = 0.0
        best_cal_r = 1.0
        best_cal_p = 0.0
        best_bound = 0.0
        found = False

        for tau in reversed(candidates):
            misses = np.sum(pos_scores < tau)
            empirical_risk = misses / n_pos
            finite_sample_bound = (n_pos / (n_pos + 1.0)) * empirical_risk + (1.0 / (n_pos + 1.0))

            if finite_sample_bound <= alpha_risk:
                rec = 1.0 - empirical_risk
                preds = s >= tau
                prec = float(np.sum((preds == 1) & pos_mask) / np.sum(preds)) if np.sum(preds) > 0 else 0.0
                best_tau = float(tau)
                best_risk = float(empirical_risk)
                best_cal_r = float(rec)
                best_cal_p = float(prec)
                best_bound = float(finite_sample_bound)
                found = True
                break

        if not found:
            misses = np.sum(pos_scores < min_threshold)
            empirical_risk = misses / n_pos
            finite_sample_bound = (n_pos / (n_pos + 1.0)) * empirical_risk + (1.0 / (n_pos + 1.0))
            best_tau = float(min_threshold)
            best_risk = float(empirical_risk)
            best_cal_r = float(1.0 - empirical_risk)
            preds = s >= min_threshold
            best_cal_p = float(np.sum((preds == 1) & pos_mask) / np.sum(preds)) if np.sum(preds) > 0 else 0.0
            best_bound = float(finite_sample_bound)

        return ConformalThresholdResult(
            target_recall=float(target_recall),
            alpha_risk=float(alpha_risk),
            fitted_threshold=float(best_tau),
            empirical_risk=float(best_risk),
            calibration_recall=float(best_cal_r),
            calibration_precision=float(best_cal_p),
            finite_sample_bound=float(best_bound),
            valid_samples=n_pos,
            guarantee_satisfied=bool(best_bound <= alpha_risk),
        )


# =========================================================================
# Conformal Safety Gate for Edge Deployment
# =========================================================================

class ConformalSafetyGate:
    """Runtime Conformal Safety Gate for edge inference.

    Applies dual-threshold safety filtering:
    1. Operational nominal threshold (e.g. tau_deploy = 0.25) for non-critical detections.
    2. Conformal Safety threshold (tau_975 or tau_95) for certified relevant red lights.
    """

    def __init__(
        self,
        tau_nominal: float = 0.25,
        tau_safety_95: float = 0.35,
        tau_safety_975: float = 0.20,
        red_prob_threshold: float = 0.40,
        calibrator: MultiTaskTemperatureCalibrator | None = None,
    ) -> None:
        self.tau_nominal = float(tau_nominal)
        self.tau_safety_95 = float(tau_safety_95)
        self.tau_safety_975 = float(tau_safety_975)
        self.red_prob_threshold = float(red_prob_threshold)
        self.calibrator = calibrator

    def evaluate_safety_gate(
        self,
        state_probs: torch.Tensor,
        relevance_probs: torch.Tensor,
        detection_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Vectorized safety evaluation on detection candidates.

        Args:
            state_probs: [B, 4, N] or [N, 4] class probabilities (0=Red, 1=Yellow, 2=Green, 3=Off)
            relevance_probs: [B, 1, N] or [N] relevance probabilities in [0, 1]
            detection_scores: [B, N] or [N] detection confidence scores

        Returns:
            Dictionary containing boolean tensor flags for safety alerts and certified actions.
        """
        if state_probs.ndim == 3:
            red_prob = state_probs[:, 0, :]
            rel_prob = relevance_probs.squeeze(1) if relevance_probs.ndim == 3 else relevance_probs
            det_score = detection_scores
        else:
            red_prob = state_probs[:, 0]
            rel_prob = relevance_probs.squeeze(-1) if relevance_probs.ndim > 1 else relevance_probs
            det_score = detection_scores

        joint_red_safety_score = red_prob * rel_prob

        is_red_candidate = red_prob >= self.red_prob_threshold
        is_safe_deploy = det_score >= self.tau_nominal
        is_safety_95_certified = (joint_red_safety_score >= self.tau_safety_95) & is_red_candidate
        is_safety_975_certified = (joint_red_safety_score >= self.tau_safety_975) & is_red_candidate

        emergency_brake_trigger = is_safety_975_certified | (is_safe_deploy & (rel_prob >= self.tau_nominal) & is_red_candidate)

        return {
            "red_probabilities": red_prob,
            "relevance_probabilities": rel_prob,
            "joint_red_safety_scores": joint_red_safety_score,
            "is_safety_95_certified": is_safety_95_certified,
            "is_safety_975_certified": is_safety_975_certified,
            "emergency_brake_trigger": emergency_brake_trigger,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_nominal": self.tau_nominal,
            "tau_safety_95": self.tau_safety_95,
            "tau_safety_975": self.tau_safety_975,
            "red_prob_threshold": self.red_prob_threshold,
            "calibrator": self.calibrator.to_dict() if self.calibrator is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalSafetyGate:
        calibrator = (
            MultiTaskTemperatureCalibrator.from_dict(data["calibrator"])
            if data.get("calibrator") is not None
            else None
        )
        return cls(
            tau_nominal=data.get("tau_nominal", 0.25),
            tau_safety_95=data.get("tau_safety_95", 0.35),
            tau_safety_975=data.get("tau_safety_975", 0.20),
            red_prob_threshold=data.get("red_prob_threshold", 0.40),
            calibrator=calibrator,
        )

