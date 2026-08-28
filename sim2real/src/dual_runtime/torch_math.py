"""Small WXYZ quaternion subset used by the deployment policy."""

from __future__ import annotations

import torch


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(quaternion), vector)


def quat_to_tan_norm(quaternion: torch.Tensor) -> torch.Tensor:
    tangent = quaternion.new_zeros(quaternion.shape[:-1] + (3,))
    normal = quaternion.new_zeros(quaternion.shape[:-1] + (3,))
    tangent[..., 0] = 1.0
    normal[..., 2] = 1.0
    return torch.cat(
        (quat_apply(quaternion, tangent), quat_apply(quaternion, normal)), dim=-1
    )
