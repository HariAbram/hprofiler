from .perf import PerfBackend
from .cuda import CUDABackend
from .opencl import OpenCLBackend
from .rocm import ROCmBackend
from .openmp import OpenMPBackend
from .likwid import LIKWIDBackend
from .mpi import MPIBackend
from .nccl import NCCLBackend
from .base import Backend

ALL_BACKENDS: dict[str, type[Backend]] = {
    "cpu": PerfBackend,
    "cuda": CUDABackend,
    "opencl": OpenCLBackend,
    "rocm": ROCmBackend,
    "openmp": OpenMPBackend,
    "likwid": LIKWIDBackend,
    "mpi": MPIBackend,
    "nccl": NCCLBackend,
}

__all__ = ["ALL_BACKENDS", "Backend",
           "PerfBackend", "CUDABackend", "OpenCLBackend", "ROCmBackend", "OpenMPBackend",
           "LIKWIDBackend", "MPIBackend", "NCCLBackend"]
