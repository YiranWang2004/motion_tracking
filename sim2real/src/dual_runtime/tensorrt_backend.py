"""Compatibility imports for existing dual/onboard launchers."""
from scalebfm.tensorrt_backend import TensorRTBackend, subprocess


def attach_tensorrt(policy, directory, files, side, **kwargs):
    if policy.scalebfm.device.type != "cpu" or policy.residual.device.type != "cpu":
        raise ValueError(
            "TensorRT worker handles CUDA; observation preprocessing must stay on CPU"
        )
    backend = TensorRTBackend(directory, files, side, **kwargs)
    policy.scalebfm.accelerated_inference = backend.scale
    policy.residual.accelerated_agent = (policy.index, backend.residual)
    return backend
