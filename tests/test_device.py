"""
FP16 peak per CUDA architecture: Pascal consumer/mobile parts (cc 6.1/6.2,
GTX 10-series, Jetson TX2) have packed-FP16 throughput roughly on par with
FP32, not the 2x of later architectures (and of Pascal's datacenter P100,
cc 6.0), so a uniform fp32_tflops * 2 would overstate their FP16 roofline
ceiling by ~2x.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.device import _cuda_fp16_ratio


class TestCudaFP16Ratio(unittest.TestCase):
    def test_pascal_consumer_and_mobile_are_not_2x(self):
        self.assertEqual(_cuda_fp16_ratio(6, 1), 1.0)  # GTX 10-series
        self.assertEqual(_cuda_fp16_ratio(6, 2), 1.0)  # Jetson TX2

    def test_pascal_datacenter_is_2x(self):
        self.assertEqual(_cuda_fp16_ratio(6, 0), 2.0)  # P100

    def test_volta_and_later_are_2x(self):
        for major, minor in ((7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0)):
            self.assertEqual(_cuda_fp16_ratio(major, minor), 2.0, msg=f"{major}.{minor}")

    def test_unknown_architecture_defaults_to_2x(self):
        # Future/unrecognized compute capabilities should not silently
        # regress to a wrong crippled value -- 2.0 (correct for everything
        # Volta-onward) is the safer default than 1.0.
        self.assertEqual(_cuda_fp16_ratio(10, 0), 2.0)


if __name__ == "__main__":
    unittest.main()
