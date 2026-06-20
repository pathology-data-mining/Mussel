from typing import List, Optional, Union

GpuDeviceId = Union[int, List[int]]


def resolve_gpu_device_id(
    gpu_device_id: Optional[GpuDeviceId] = None,
    gpu_device_ids: Optional[List[int]] = None,
) -> Optional[GpuDeviceId]:
    """Apply the common precedence for single- and multi-GPU CLI options."""
    return gpu_device_ids if gpu_device_ids else gpu_device_id


def first_gpu_device_id(gpu_device_id: Optional[GpuDeviceId] = None) -> Optional[int]:
    """Return the first concrete GPU id from a scalar or multi-GPU value."""
    if isinstance(gpu_device_id, list):
        return gpu_device_id[0] if gpu_device_id else None
    return gpu_device_id
