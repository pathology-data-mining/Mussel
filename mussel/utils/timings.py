"""Opt-in stage timings, written as JSON lines.

The CLIs take ``timings_path=<file>``; while it is set, each stage (model load,
one slide's patch features, one slide's tessellation, ...) appends one JSON
object to the file when it ends::

    {"stage": "patch_features", "model": "HOPTIMUS1", "slide": "a.svs", "tiles": 41216,
     "start": 1791830000.1, "end": 1791830061.4, "seconds": 61.3,
     "load_wait_s": 2.1, "model_s": 57.9, "write_s": 0.8, "batches": 644,
     "tiles_done": 41216, "gpu_mem_peak_mib": 9421.0}

``start``/``end`` are epoch seconds, so the records can be lined up with
timings taken outside the process (e.g. a workflow manager's task start).
Counters such as ``load_wait_s`` (time blocked waiting for the next batch from
the DataLoader) and ``model_s`` (forward pass, including the copy back to the
CPU) are summed over the stage; ``tiles_done`` counts the tiles that went
through the model (fewer than ``tiles`` in a test run). ``gpu_mem_peak_mib`` is the peak memory
PyTorch allocated on the GPU during the stage.

When no file is set every call here is a no-op, and :func:`timed_iter`
returns its iterable unchanged.

Caveats while recording:

- GPU peaks: each stage calls ``torch.cuda.reset_peak_memory_stats()``, which is
  process-wide, so other code reading ``torch.cuda.max_memory_allocated()``
  sees peaks since the last stage started, not since its own reset. Only the
  current CUDA device is measured (multi-GPU runs under-report).
- Threads: the stage stack is module state; record from one thread only.
  DataLoader worker processes import this module afresh and record nothing.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

_out = None  # open file while recording
_stack: list[dict] = []  # open stages, innermost last


@contextmanager
def recording(path: Optional[str]):
    """Append stage timings to ``path`` within this block (no-op if ``path`` is empty)."""
    global _out
    if not path:
        yield
        return
    previous = _out
    _out = open(path, "a", buffering=1)
    try:
        yield
    finally:
        _out.close()
        _out = previous


def enabled() -> bool:
    return _out is not None


def _cuda():
    """torch.cuda if this process has initialized CUDA, else None (never initializes it)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return torch.cuda


@contextmanager
def stage(name: str, **fields):
    """Time a stage; yields its record (a dict, or None when not recording).

    The record is written when the block ends, also when it raises (then with
    ``"error"`` set to the exception type).
    """
    if _out is None:
        yield None
        return
    cuda = _cuda()
    if cuda is not None:
        # The enclosing stage keeps its own peak so far; this stage measures from here.
        if _stack:
            _stack[-1]["_peak"] = max(_stack[-1].get("_peak", 0), cuda.max_memory_allocated())
        cuda.reset_peak_memory_stats()
    record = {"stage": name, **fields, "start": time.time()}
    _stack.append(record)
    try:
        yield record
    except BaseException as exc:
        record["error"] = type(exc).__name__
        raise
    finally:
        _stack.pop()
        record["end"] = time.time()
        record["seconds"] = round(record["end"] - record["start"], 3)
        peak = record.pop("_peak", 0)
        cuda = _cuda()
        if cuda is not None:
            peak = max(peak, cuda.max_memory_allocated())
            record["gpu_mem_peak_mib"] = round(peak / 2**20, 1)
            if _stack:
                _stack[-1]["_peak"] = max(_stack[-1].get("_peak", 0), peak)
        for key, value in record.items():
            if key.endswith("_s") and isinstance(value, float):
                record[key] = round(value, 3)
        if _out is not None:
            _out.write(json.dumps(record, default=str) + "\n")


def add(**amounts) -> None:
    """Add to counters of the innermost open stage (e.g. ``add(model_s=0.4, batches=1)``)."""
    if _stack:
        record = _stack[-1]
        for key, value in amounts.items():
            record[key] = record.get(key, 0) + value


@contextmanager
def counting(key: str):
    """Add the block's duration to counter ``key`` of the innermost open stage."""
    if not _stack:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add(**{key: time.perf_counter() - t0})


def timed_iter(iterable: Iterable, key: str = "load_wait_s") -> Iterable:
    """``iterable``, adding the time spent waiting for each item to counter ``key``."""
    if _out is None:
        return iterable
    return _timed_iter(iter(iterable), key)


def _timed_iter(it: Iterator, key: str) -> Iterator:
    while True:
        t0 = time.perf_counter()
        try:
            item = next(it)
        except StopIteration:
            return
        add(**{key: time.perf_counter() - t0})
        yield item
