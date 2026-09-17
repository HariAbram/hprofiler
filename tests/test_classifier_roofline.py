"""
Regression tests for a self-audit bug: integer address-arithmetic
instructions (SASS IMAD/IMUL/XMAD, PTX mad.lo.s32-style, AMDGCN
v_add_u32-style) were classified into the same InsnType.COMPUTE bucket as
real floating-point ops and charged FLOPs in the disasm-based roofline
estimate -- inflating achieved_tflops/arithmetic_intensity and able to flip
the compute-vs-memory-bound verdict for a purely memory-bound kernel with
normal address arithmetic. Tensor-core ops (HMMA/BMMA, MFMA) were also
lumped into the same 2-FLOPs-per-instruction bucket as a scalar FMA,
undercounting real tensor-core throughput by ~100x+.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.disasm.classifier import (
    InsnType, classify_sass, classify_ptx, classify_amdgcn,
)
from src.disasm.extractor import DisasmLine, KernelDisasm
from src.analysis.device import DevicePeak
from src.analysis.roofline import compute_kernel_metrics, _FLOPS
from src.core.events import SpanEvent, Category


def _device() -> DevicePeak:
    return DevicePeak(
        name="test-gpu", backend="cuda", fp32_tflops=10.0, fp64_tflops=5.0,
        fp16_tflops=20.0, bandwidth_gbs=500.0, sm_count=80, core_clock_ghz=1.5,
        mem_clock_ghz=1.0, mem_bus_bits=4096, vram_gb=40.0, compute_cap="8.0",
    )


class TestSASSClassification(unittest.TestCase):
    def test_int_mad_is_not_compute(self):
        self.assertEqual(classify_sass("IMAD"), InsnType.INT_COMPUTE)
        self.assertEqual(classify_sass("IMUL"), InsnType.INT_COMPUTE)
        self.assertEqual(classify_sass("XMAD"), InsnType.INT_COMPUTE)
        self.assertNotEqual(classify_sass("IMAD"), InsnType.COMPUTE)

    def test_real_fp_ops_still_compute(self):
        for m in ("FFMA", "DFMA", "HFMA", "FMUL", "FADD", "FDIV", "DMUL", "DADD"):
            self.assertEqual(classify_sass(m), InsnType.COMPUTE, msg=m)

    def test_tensor_core_ops_are_tensor(self):
        for m in ("HMMA", "BMMA", "IMMA", "DMMA"):
            self.assertEqual(classify_sass(m), InsnType.TENSOR, msg=m)


class TestPTXClassification(unittest.TestCase):
    def test_integer_mad_by_type_suffix(self):
        self.assertEqual(classify_ptx("mad.lo.s32"), InsnType.INT_COMPUTE)
        self.assertEqual(classify_ptx("mad.lo.s64"), InsnType.INT_COMPUTE)
        self.assertEqual(classify_ptx("mul.wide.u32"), InsnType.INT_COMPUTE)
        self.assertEqual(classify_ptx("add.s64"), InsnType.INT_COMPUTE)

    def test_fp_fma_by_type_suffix(self):
        self.assertEqual(classify_ptx("fma.rn.f32"), InsnType.COMPUTE)
        self.assertEqual(classify_ptx("mad.rn.f64"), InsnType.COMPUTE)
        self.assertEqual(classify_ptx("add.f32"), InsnType.COMPUTE)

    def test_tensor_ops(self):
        self.assertEqual(classify_ptx("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
                          InsnType.TENSOR)
        self.assertEqual(classify_ptx("wmma.mma.sync.aligned"), InsnType.TENSOR)


class TestAMDGCNClassification(unittest.TestCase):
    def test_integer_lane_ops_not_charged(self):
        self.assertEqual(classify_amdgcn("v_add_u32"), InsnType.INT_COMPUTE)
        self.assertEqual(classify_amdgcn("v_mad_i32"), InsnType.INT_COMPUTE)

    def test_real_fp_ops_unaffected(self):
        self.assertEqual(classify_amdgcn("v_fma_f32"), InsnType.VEC_SP)
        self.assertEqual(classify_amdgcn("v_fma_f64"), InsnType.VEC_DP)

    def test_mfma_is_tensor_not_vec_sp(self):
        # Contains "_f32" as a substring -- must be caught by the MFMA
        # check BEFORE the generic "_f32" -> VEC_SP check.
        self.assertEqual(classify_amdgcn("v_mfma_f32_32x32x8f16"), InsnType.TENSOR)


class TestFlopsTableEntries(unittest.TestCase):
    def test_int_compute_is_zero_everywhere_defined(self):
        for arch, table in _FLOPS.items():
            if InsnType.INT_COMPUTE in table:
                self.assertEqual(table[InsnType.INT_COMPUTE], 0.0, msg=arch)

    def test_tensor_is_much_higher_than_scalar_fma(self):
        self.assertGreater(_FLOPS["sass"][InsnType.TENSOR], 50 * _FLOPS["sass"][InsnType.COMPUTE])
        self.assertGreater(_FLOPS["amdgcn"][InsnType.TENSOR], 50 * _FLOPS["amdgcn"][InsnType.COMPUTE])


class TestEndToEndKernelMetrics(unittest.TestCase):
    def test_pure_address_arithmetic_kernel_has_zero_flops(self):
        """A purely memory-bound kernel (loads + integer address math, no
        real FP ops) must estimate est_flops == 0 -- before the fix, the
        IMAD instructions here would have been charged 2.0 FLOPs each."""
        lines = [
            DisasmLine(mnemonic="IMAD", operands="R1, R2, R3, R4", itype=classify_sass("IMAD")),
            DisasmLine(mnemonic="IMAD", operands="R5, R6, R3, R7", itype=classify_sass("IMAD")),
            DisasmLine(mnemonic="LDG", operands="R8, [R1]", itype=classify_sass("LDG")),
            DisasmLine(mnemonic="STG", operands="[R5], R8", itype=classify_sass("STG")),
        ]
        kd = KernelDisasm(name="copy_kernel", arch="sass", source="", lines=lines)
        span = SpanEvent(name="copy_kernel", category=Category.GPU_CUDA,
                          start_ns=0, duration_ns=1_000_000,
                          tags={"grid": "1x1x1", "block": "256x1x1"})
        metrics = compute_kernel_metrics(span, kd, _device())
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.est_flops, 0.0)
        self.assertEqual(metrics.bound, "memory")  # bytes>0, ai=0 < ridge -> memory-bound

    def test_real_fp_kernel_has_nonzero_flops(self):
        lines = [
            DisasmLine(mnemonic="FFMA", operands="R1, R2, R3, R4", itype=classify_sass("FFMA")),
            DisasmLine(mnemonic="FFMA", operands="R1, R2, R3, R4", itype=classify_sass("FFMA")),
        ]
        kd = KernelDisasm(name="fma_kernel", arch="sass", source="", lines=lines)
        span = SpanEvent(name="fma_kernel", category=Category.GPU_CUDA,
                          start_ns=0, duration_ns=1_000_000,
                          tags={"grid": "1x1x1", "block": "256x1x1"})
        metrics = compute_kernel_metrics(span, kd, _device())
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.est_flops, 0.0)


if __name__ == "__main__":
    unittest.main()
