"""Pure-PyTorch batch ScaleBFM runtime without mjlab/IsaacLab imports."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import KEY_BODY_NAMES, NUM_JOINTS, POLICY_JOINT_NAMES
from .scalebfm_network import HumanoidTransformer, TaskEmbedder
from .torch_math import quat_apply_inverse, quat_conjugate, quat_mul, quat_to_tan_norm


class ScaleBFMHistory:
    """Three-frame state/action history matching ScaleBridge reset semantics."""

    def __init__(self, length: int = 3):
        self.length = int(length)
        self._values: deque[tuple[np.ndarray, ...]] = deque(maxlen=self.length)

    def reset(self) -> None:
        self._values.clear()

    def update(
        self,
        root_quat_wxyz: np.ndarray,
        base_ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        previous_executed_action: np.ndarray,
    ) -> None:
        values = tuple(
            np.asarray(value, dtype=np.float32).reshape(size).copy()
            for value, size in (
                (root_quat_wxyz, 4),
                (base_ang_vel, 3),
                (joint_pos, NUM_JOINTS),
                (joint_vel, NUM_JOINTS),
                (previous_executed_action, NUM_JOINTS),
            )
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("ScaleBFM history contains non-finite values")
        if not self._values:
            zero_action = np.zeros(NUM_JOINTS, dtype=np.float32)
            for _ in range(self.length):
                self._values.append((*values[:4], zero_action.copy()))
        else:
            self._values.append(values)

    def arrays(self) -> tuple[np.ndarray, ...]:
        if len(self._values) != self.length:
            raise RuntimeError("ScaleBFM history is not initialized")
        return tuple(
            np.stack([row[index] for row in self._values]) for index in range(5)
        )


class _EagerPolicy(nn.Module):
    def __init__(self, actor: nn.Module, embedder: nn.Module):
        super().__init__()
        self.actor = actor
        self.task_embedder = embedder

    def forward(
        self, prop: torch.Tensor, actions: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        return self.actor(prop, actions, self.task_embedder(task))


class ScaleBFMPolicy:
    """Two-robot batched ScaleBFM target generator."""

    def __init__(
        self,
        checkpoint: str | Path,
        metadata: str | Path,
        mode_table: str | Path,
        *,
        device: str = "cpu",
        inference_precision: str = "fp32",
        torch_num_threads: int | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.metadata_path = Path(metadata).expanduser().resolve()
        self.mode_table_path = Path(mode_table).expanduser().resolve()
        for path in (self.checkpoint, self.metadata_path, self.mode_table_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.metadata: dict[str, Any] = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )
        self._validate_metadata()
        self.device = torch.device(device)
        if torch_num_threads is not None:
            if int(torch_num_threads) < 1:
                raise ValueError("torch_num_threads must be positive")
            torch.set_num_threads(int(torch_num_threads))
        if inference_precision not in {"fp32", "tf32", "fp16"}:
            raise ValueError("inference_precision must be fp32, tf32, or fp16")
        if inference_precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 ScaleBFM inference requires CUDA")
        self.inference_precision = inference_precision
        self.mode_table = torch.load(
            self.mode_table_path, map_location=self.device, weights_only=True
        ).to(dtype=torch.float32)
        if tuple(self.mode_table.shape) != (8, len(KEY_BODY_NAMES)):
            raise ValueError("ScaleBFM mode table must have shape (8, 14)")
        self.policy = self._load_policy()
        self.default_q = np.asarray(self.metadata["default_dof_pos"], dtype=np.float32)
        self.action_scale = np.asarray(self.metadata["action_scale"], dtype=np.float32)
        self.kp = np.asarray(self.metadata["stiffness"], dtype=np.float32)
        self.kd = np.asarray(self.metadata["damping"], dtype=np.float32)
        self.torque_limit = np.asarray(self.metadata["torque_limit"], dtype=np.float32)
        self._default_q = torch.as_tensor(self.default_q, device=self.device)[None]
        self._action_scale = torch.as_tensor(self.action_scale, device=self.device)[
            None
        ]
        self._gravity = torch.tensor(
            (0.0, 0.0, -1.0), dtype=torch.float32, device=self.device
        )
        mappings = []
        for mode_vector in self.mode_table:
            mappings.append(
                torch.cat(
                    tuple(
                        mode_vector.unsqueeze(-1).expand(-1, dim).flatten()
                        for dim in self.metadata["mode_feature_dims"]
                    )
                    + (torch.ones(1, dtype=torch.float32, device=self.device),)
                )
            )
        self._mode_mapping_table = torch.stack(mappings)

    def _validate_metadata(self) -> None:
        if tuple(self.metadata.get("joint_names", ())) != POLICY_JOINT_NAMES:
            raise ValueError("ScaleBFM joint order does not match bridge policy order")
        if tuple(self.metadata.get("action_names", ())) != POLICY_JOINT_NAMES:
            raise ValueError("ScaleBFM action order does not match bridge policy order")
        if tuple(self.metadata.get("selected_body_names", ())) != KEY_BODY_NAMES:
            raise ValueError("ScaleBFM selected body order is unsupported")
        if int(self.metadata.get("history_buffer_size", -1)) != 3:
            raise ValueError("ScaleBFM history_buffer_size must be 3")
        for name in (
            "default_dof_pos",
            "action_scale",
            "stiffness",
            "damping",
            "torque_limit",
        ):
            value = np.asarray(self.metadata.get(name), dtype=np.float32)
            if value.shape != (NUM_JOINTS,) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid ScaleBFM metadata field {name}")

    def _load_policy(self) -> nn.Module:
        architecture = self.metadata["policy_architecture"]
        actor = HumanoidTransformer(
            prop_obs_dim=int(architecture["prop_obs_dim"]),
            action_dim=int(architecture["action_dim"]),
            output_dim=int(architecture["output_dim"]),
            embed_dim=int(architecture["embedding_dim"]),
            num_heads=int(architecture["num_heads"]),
            ff_dim=int(architecture["ff_dim"]),
            num_layers=int(architecture["num_layers"]),
        ).to(self.device)
        embedder = TaskEmbedder(
            task_obs_dim=int(architecture["task_obs_dim"]),
            embedding_dim=int(architecture["embedding_dim"]),
            reduced_task_dim=architecture.get("reduced_task_dim"),
            hidden_dims=architecture.get("task_embedder_hidden_dims", []),
        ).to(self.device)
        checkpoint = torch.load(
            self.checkpoint, map_location=self.device, weights_only=True
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        actor.load_state_dict(
            {
                key.removeprefix("actor."): value
                for key, value in state.items()
                if key.startswith("actor.")
            },
            strict=True,
        )
        embedder.load_state_dict(
            {
                key.removeprefix("actor_task_embedder."): value
                for key, value in state.items()
                if key.startswith("actor_task_embedder.")
            },
            strict=True,
        )
        return (
            _EagerPolicy(actor, embedder).requires_grad_(False).eval().to(self.device)
        )

    def _run_policy(
        self, prop: torch.Tensor, actions: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        accelerated = getattr(self, "accelerated_inference", None)
        if accelerated is not None:
            return accelerated(prop, actions, task)
        if self.inference_precision == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.policy(prop, actions, task).float()
        if self.device.type == "cuda":
            previous = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = (
                    self.inference_precision == "tf32"
                )
                return self.policy(prop, actions, task)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        return self.policy(prop, actions, task)

    def infer_batch(
        self,
        histories: tuple[ScaleBFMHistory, ...],
        target_body_pos_w: np.ndarray,
        target_body_quat_wxyz: np.ndarray,
        live_body_pos_w: np.ndarray,
        live_body_quat_wxyz: np.ndarray,
        *,
        control_mode: int,
        time_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= int(control_mode) < 8:
            raise ValueError("control_mode must be in [0, 7]")
        batch = len(histories)
        if batch < 1:
            raise ValueError("at least one ScaleBFM history is required")
        history_arrays = [history.arrays() for history in histories]
        root_quat, base_ang_vel, joint_pos, joint_vel, actions = (
            torch.as_tensor(
                np.stack([values[index] for values in history_arrays]),
                dtype=torch.float32,
                device=self.device,
            )
            for index in range(5)
        )
        target_pos = torch.as_tensor(
            target_body_pos_w, dtype=torch.float32, device=self.device
        ).reshape(batch, 6, len(KEY_BODY_NAMES), 3)
        target_quat = torch.as_tensor(
            target_body_quat_wxyz, dtype=torch.float32, device=self.device
        ).reshape(batch, 6, len(KEY_BODY_NAMES), 4)
        live_pos = torch.as_tensor(
            live_body_pos_w, dtype=torch.float32, device=self.device
        ).reshape(batch, len(KEY_BODY_NAMES), 3)
        live_quat = torch.as_tensor(
            live_body_quat_wxyz, dtype=torch.float32, device=self.device
        ).reshape(batch, len(KEY_BODY_NAMES), 4)
        offsets = (
            torch.as_tensor(time_offsets, dtype=torch.float32, device=self.device)
            .reshape(1, 6, 1)
            .expand(batch, -1, -1)
        )
        tensors = (root_quat, base_ang_vel, joint_pos, joint_vel, actions)
        if not all(
            torch.isfinite(value).all()
            for value in (*tensors, target_pos, target_quat, live_pos, live_quat)
        ):
            raise ValueError("ScaleBFM inference inputs contain non-finite values")

        pelvis_pos = live_pos[:, 0]
        pelvis_quat = live_quat[:, 0]
        pelvis_quat_future = pelvis_quat[:, None, None].expand_as(target_quat)
        target_pos_base = quat_apply_inverse(
            pelvis_quat_future, target_pos - pelvis_pos[:, None, None]
        )
        target_rot_base = quat_mul(quat_conjugate(pelvis_quat_future), target_quat)
        live_pos_future = live_pos[:, None].expand_as(target_pos)
        target_pos_rel = quat_apply_inverse(
            pelvis_quat_future, target_pos - live_pos_future
        )
        live_quat_future = live_quat[:, None].expand_as(target_quat)
        target_rot_rel = quat_mul(target_quat, quat_conjugate(live_quat_future))
        target_rot_rel = quat_mul(
            quat_mul(quat_conjugate(pelvis_quat_future), target_rot_rel),
            pelvis_quat_future,
        )
        task = torch.cat(
            (
                target_pos_base.flatten(2),
                target_pos_rel.flatten(2),
                quat_to_tan_norm(target_rot_base).flatten(2),
                quat_to_tan_norm(target_rot_rel).flatten(2),
                offsets,
            ),
            dim=-1,
        )
        modes = torch.full(
            (batch,), int(control_mode), dtype=torch.long, device=self.device
        )
        task = task * self._mode_mapping_table[modes, None]
        mode_vector = self.mode_table[modes]
        task = torch.cat(
            (task, mode_vector[:, None].expand(-1, task.shape[1], -1)), dim=-1
        )
        projected_gravity = quat_apply_inverse(
            root_quat,
            self._gravity.expand(root_quat.shape[0], root_quat.shape[1], 3),
        )
        prop = torch.cat(
            (
                projected_gravity,
                base_ang_vel,
                joint_pos - self._default_q[:, None],
                joint_vel * 0.05,
            ),
            dim=-1,
        )
        with torch.inference_mode():
            raw_action = self._run_policy(prop, actions, task)
            target = raw_action * self._action_scale + self._default_q
        if not torch.isfinite(target).all() or not torch.isfinite(raw_action).all():
            raise RuntimeError("ScaleBFM produced non-finite output")
        return (
            target.detach().cpu().numpy().astype(np.float32),
            raw_action.detach().cpu().numpy().astype(np.float32),
        )
