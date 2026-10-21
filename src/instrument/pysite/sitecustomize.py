"""
hprofiler's Python-side JIT instrumentation (`hprofiler run --python-jit`).

This directory is put first on PYTHONPATH, so Python imports this module at
start-up in place of a sitecustomize further down the path. It runs that one
right away (the program's own, the distribution's), so nothing changes for
the program, and leaves an import hook behind: when the program imports a
Python JIT compiler it knows (Triton), _hprofiler_pyjit registers its
compilation listener. Nothing else happens in a Python program that does not
import one, and nothing at all in other programs.

Only this file and _hprofiler_pyjit.py may live in this directory: anything
else here would shadow the program's own modules.
"""
import os
import sys


def _run_next_sitecustomize() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ours = sys.modules.pop("sitecustomize", None)
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    try:
        import sitecustomize  # noqa: F401 -- the next one on the path, if any
    except ImportError as e:
        if e.name != "sitecustomize":
            raise
    except Exception:
        # theirs failed: Python would only print it; so do we
        import traceback
        traceback.print_exc()
    finally:
        sys.path[:] = saved
        if "sitecustomize" not in sys.modules and ours is not None:
            sys.modules["sitecustomize"] = ours


_run_next_sitecustomize()

if os.environ.get("HPROFILER_PYTHON_JIT") == "1" and os.environ.get("HPROFILER_SOCKET"):
    try:
        import _hprofiler_pyjit           # next to this file (first on PYTHONPATH)
        _hprofiler_pyjit.install()
    except Exception as e:                 # never break the program
        print(f"[hprofiler] python-jit instrumentation disabled: {e!r}", file=sys.stderr)
