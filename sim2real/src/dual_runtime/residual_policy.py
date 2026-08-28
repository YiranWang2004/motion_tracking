"""Deterministic MAPPO residual Actor runtime with frozen preprocessors."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _ResidualActor(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        widths = (201, 512, 256, 128, 29)
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(pairwise(widths)):
            linear = nn.Linear(input_width, output_width)
            weight_key = f"net_container.{2 * index}.weight"
            bias_key = f"net_container.{2 * index}.bias"
            weight = state.get(weight_key)
            bias = state.get(bias_key)
            if weight is None or bias is None:
                raise ValueError(
                    f"residual checkpoint is missing {weight_key}/{bias_key}"
                )
            if tuple(weight.shape) != (output_width, input_width):
                raise ValueError(f"unexpected residual layer shape for {weight_key}")
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            if index < 3:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class ResidualPolicy:
    """Load robot_0/robot_1 Actor means and RunningStandardScaler state."""

    AGENTS = ("robot_0", "robot_1")

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = torch.device(device)
        modules = torch.load(
            self.checkpoint, map_location=self.device, weights_only=True
        )
        self.actors: list[_ResidualActor] = []
        means = []
        variances = []
        for agent in self.AGENTS:
            entry = modules.get(agent)
            if not isinstance(entry, dict):
                raise TypeError(f"residual checkpoint is missing {agent}")
            policy = entry.get("policy")
            preprocessor = entry.get("state_preprocessor")
            if not isinstance(policy, dict) or not isinstance(preprocessor, dict):
                raise TypeError(
                    f"residual checkpoint is missing {agent} inference state"
                )
            actor = _ResidualActor(policy).requires_grad_(False).eval().to(self.device)
            mean = preprocessor.get("running_mean")
            variance = preprocessor.get("running_variance")
            if (
                mean is None
                or variance is None
                or tuple(mean.shape) != (201,)
                or tuple(variance.shape) != (201,)
            ):
                raise ValueError(f"invalid {agent} RunningStandardScaler")
            if torch.any(variance < 0.0):
                raise ValueError(f"negative {agent} running variance")
            self.actors.append(actor)
            means.append(mean.to(device=self.device, dtype=torch.float32))
            variances.append(variance.to(device=self.device, dtype=torch.float32))
        self.means = torch.stack(means)
        self.variances = torch.stack(variances)

    def infer(self, observations: np.ndarray) -> np.ndarray:
        value = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).reshape(2, 201)
        if not torch.isfinite(value).all():
            raise ValueError("residual observations contain non-finite values")
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0).clamp(
            -100.0, 100.0
        )
        normalized = (
            (value - self.means) / (torch.sqrt(self.variances) + 1.0e-8)
        ).clamp(-5.0, 5.0)
        with torch.inference_mode():
            action = torch.stack(
                [self.actors[index](normalized[index]) for index in range(2)]
            ).clamp(-1.0, 1.0)
        if not torch.isfinite(action).all():
            raise RuntimeError("residual Actor produced non-finite output")
        return action.cpu().numpy().astype(np.float32)
