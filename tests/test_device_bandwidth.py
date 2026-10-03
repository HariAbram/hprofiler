"""
Peak DRAM bandwidth from the driver's memory clock / bus width
(src/analysis/device.dram_bandwidth_gbs) for representative devices, with
the attribute values their drivers report, against vendor datasheets; plus
provenance (computed / datasheet / unavailable) and its round trip through
saved traces.

The x2 data-rate factor that is right for every NVIDIA part and AMD up to
CDNA2 under-reported AMD CDNA3 (MI300, HBM3: the reported 1300 MHz clock is
a quarter of the 5.2 Gb/s pin rate) by half.

Only the MX550 values were read from real hardware on the development
machine; the others are the drivers' documented attribute values, so these
tests check the formula, not those drivers.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.device import DevicePeak, dram_bandwidth_gbs

# (name, backend, arch, MEMORY_CLOCK_RATE kHz, bus bits, datasheet GB/s)
DEVICES = [
    ("Tesla V100-SXM2-16GB", "cuda", "7.0", 877_000, 4096, 900.0),
    ("A100-SXM4-40GB", "cuda", "8.0", 1_215_000, 5120, 1555.0),
    ("A100-SXM4-80GB", "cuda", "8.0", 1_593_000, 5120, 2039.0),
    ("H100 80GB HBM3 (SXM)", "cuda", "9.0", 2_619_000, 5120, 3350.0),
    ("NVIDIA GeForce RTX 4090", "cuda", "8.9", 10_501_000, 384, 1008.0),
    ("NVIDIA GeForce MX550", "cuda", "7.5", 6_001_000, 64, 96.0),
    ("AMD Instinct MI100", "rocm", "gfx908", 1_200_000, 4096, 1228.8),
    ("AMD Instinct MI210", "rocm", "gfx90a", 1_600_000, 4096, 1638.4),
    ("AMD Instinct MI250X", "rocm", "gfx90a", 1_600_000, 8192, 3276.8),
    ("AMD Instinct MI300X", "rocm", "gfx942", 1_300_000, 8192, 5300.0),
    ("AMD Instinct MI300A", "rocm", "gfx942", 1_300_000, 8192, 5300.0),
]


class TestDramBandwidth(unittest.TestCase):
    def test_representative_devices_match_datasheets(self):
        for name, backend, arch, clk, bus, datasheet in DEVICES:
            with self.subTest(name):
                gbs, src = dram_bandwidth_gbs(backend, arch, clk, bus, name)
                self.assertEqual(src, "computed")
                self.assertLess(abs(gbs - datasheet) / datasheet, 0.02,
                                f"{name}: computed {gbs:.0f} GB/s vs datasheet {datasheet:.0f}")

    def test_cdna3_no_longer_halved(self):
        gbs, _ = dram_bandwidth_gbs("rocm", "gfx942", 1_300_000, 8192)
        self.assertGreater(gbs, 5000)
        # the same clock on a CDNA2 part keeps the x2 factor
        self.assertAlmostEqual(dram_bandwidth_gbs("rocm", "gfx90a", 1_300_000, 8192)[0], 2662.4, places=1)

    def test_datasheet_only_when_attributes_missing(self):
        gbs, src = dram_bandwidth_gbs("rocm", "gfx942", 0, 8192, "AMD Instinct MI300X")
        self.assertEqual((gbs, src), (5300.0, "datasheet"))
        gbs, src = dram_bandwidth_gbs("cuda", "9.0", 2_619_000, 0, "NVIDIA H100 PCIe")
        self.assertEqual((gbs, src), (2000.0, "datasheet"))
        self.assertEqual(dram_bandwidth_gbs("cuda", "9.9", 0, 0, "Unknown GPU"), (0.0, "unavailable"))

    def test_provenance_round_trip_and_old_traces(self):
        d = DevicePeak(name="x", backend="cuda", fp32_tflops=1, fp64_tflops=0, fp16_tflops=0,
                       bandwidth_gbs=96, sm_count=1, core_clock_ghz=1, mem_clock_ghz=6, mem_bus_bits=64,
                       vram_gb=2, compute_cap="7.5", provenance={"bandwidth_gbs": "computed"})
        back = DevicePeak.from_dict(d.to_dict())
        self.assertEqual(back.source_of("bandwidth_gbs"), "computed")
        old = d.to_dict()
        del old["provenance"]                       # a trace saved before provenance existed
        self.assertEqual(DevicePeak.from_dict(old).source_of("bandwidth_gbs"), "")

    def test_provenance_survives_a_saved_trace(self):
        import tempfile, shutil
        from src.core import trace_io
        from src.core.trace import TraceMetadata
        tmp = Path(tempfile.mkdtemp(prefix="hp_dev_"))
        try:
            t = trace_io.create_disk_trace(tmp / "d.hpstore", TraceMetadata())
            t.set_devices([DevicePeak(name="MI300X", backend="rocm", fp32_tflops=1, fp64_tflops=1,
                                      fp16_tflops=1, bandwidth_gbs=5324.8, sm_count=304,
                                      core_clock_ghz=2.1, mem_clock_ghz=1.3, mem_bus_bits=8192,
                                      vram_gb=192, compute_cap="gfx942",
                                      provenance={"bandwidth_gbs": "computed",
                                                  "l2_bandwidth_gbs": "estimate"})])
            t.finalize()
            t.save()
            t.close()
            r = trace_io.open_trace(tmp / "d.hpstore")
            self.assertEqual(r.devices[0].source_of("bandwidth_gbs"), "computed")
            self.assertEqual(r.devices[0].source_of("l2_bandwidth_gbs"), "estimate")
            r.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
