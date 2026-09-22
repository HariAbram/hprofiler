"""
Regression test for src/core/runner.py's _total_zero_event_warning() --
fires when a run completes but captures zero events across EVERY active
backend, a much stronger signal than any one backend's own zero-event
check that the hooks never connected to the collector socket at all for
that run.

A real user hit exactly this profiling GROMACS via `srun` (SLURM): the
run completed normally (full GROMACS performance summary printed) but
captured zero spans of any kind across both active backends (mpi,
openmp), then the IDENTICAL command captured 60381 events on the very
next invocation with no code change in between -- consistent with `srun`
not propagating HPROFILER_SOCKET/LD_PRELOAD to the spawned job step on
that particular invocation (a launcher/site environment-export issue,
not a hprofiler hook race: every hook's ensure_connected() retries on
every single emit call, so a transient "listener not ready yet" race
would only lose the first few events, not literally all of them across a
24-second run).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.runner import _total_zero_event_warning


class TestTotalZeroEventWarning(unittest.TestCase):
    def test_names_all_active_backends(self):
        msg = _total_zero_event_warning(["mpi", "openmp"], ["./app"])
        self.assertIn("mpi, openmp", msg)
        self.assertIn("ZERO events", msg)

    def test_srun_gets_a_launcher_specific_hint(self):
        msg = _total_zero_event_warning(
            ["mpi", "openmp"], ["srun", "-n", "1", "gmx_mpi", "mdrun"])
        self.assertIn("job launcher", msg)
        self.assertIn("srun --export=ALL", msg)
        self.assertIn("propagate", msg)

    def test_mpirun_also_recognized_as_a_launcher(self):
        msg = _total_zero_event_warning(["mpi"], ["mpirun", "-np", "4", "./app"])
        self.assertIn("job launcher", msg)
        self.assertIn("mpirun --export=ALL", msg)

    def test_plain_binary_gets_no_launcher_hint(self):
        # A direct binary invocation (no srun/mpirun/... wrapper) --
        # the launcher-specific advice would be actively wrong here, so
        # it must not appear.
        msg = _total_zero_event_warning(["openmp"], ["./my_program"])
        self.assertNotIn("job launcher", msg)
        self.assertIn("ZERO events", msg)

    def test_empty_command_does_not_crash(self):
        msg = _total_zero_event_warning(["cpu"], [])
        self.assertIn("ZERO events", msg)


if __name__ == "__main__":
    unittest.main()
