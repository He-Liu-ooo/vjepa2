"""Lightweight GPU timing helper using torch.cuda.Event.

This module provides CudaTimerCollection which records start/end
`torch.cuda.Event` pairs and flushes elapsed times to a jsonl file in
batches. If `torch.cuda.Event.elapsed_time` is unavailable for a pair
(e.g. events not completed or other failure), the code falls back to a
CPU-based duration and the output record is annotated with
`"timing_source": "cpu_fallback"` so downstream analysis can
filter or treat those entries specially.

Usage example:
    from cosmos_predict2._src.imaginaire.utils.cuda_timer import CudaTimerCollection
    timer = CudaTimerCollection(rank=0)

    # inside critical region
    with timer.measure(action=0, step=1, block=2, layer='ffn'):
        out = ffn_layer(x)

    # periodic flush (e.g. at end of action chunk)
    timer.flush_to_file(f"outputs/timings_rank{timer.rank}.jsonl")

Notes:
- This implementation batch-synchronizes once during flush (when
  `batch_sync=True`) to avoid repeated device synchronization that would
  distort GPU timing.
- Records that used CPU fallback are annotated with
  `"timing_source": "cpu_fallback"`.
"""

import contextlib
import os
import time

import torch

import json, threading
from typing import Dict, Any, List, Tuple

class CudaTimerCollection:
    def __init__(self, rank: int = 0):
        self.rank = rank  # for distributed setting
        self._lock = threading.Lock()
        self._measures: List[Dict[str, Any]] = []  # buffer of {meta, start, end events, ts_cpu}
        self._pending_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event, Dict[str, Any]]] = []

    @contextlib.contextmanager # enable writing with timer.measure(...)
    def measure(self, **meta):
        """Context manager to time a GPU region.

        Example meta keys: action, step, block, layer, extra tags.
        """
        # create events on the current CUDA device
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        cpu_ts = time.time() # for backup or comparison, in case GPU elapse fail
        # record start on default stream (non-blocking)
        start.record()
        try:
            yield # execute the code block being wrapped
        finally:
            end.record()
            with self._lock:
                self._pending_pairs.append((start, end, {"cpu_ts": cpu_ts, **meta}))
                
    def _drain_pending(self):
        with self._lock:
            pairs = self._pending_pairs
            self._pending_pairs = []
        return pairs
    
    def flush_to_file(self, path: str, batch_sync: bool = True, use_file_lock: bool = False) -> int:
        """Compute elapsed times for pending pairs and append to `path` as jsonl.

        If `batch_sync=True` we call `torch.cuda.synchronize()` once to ensure
        all events finished; this reduces synchronization overhead compared to
        synchronizing per-event.

        Returns the number of records written.
        """
        pairs = self._drain_pending()
        if not pairs:
            return 0

        if batch_sync:
            # single synchronization to ensure events completed
            torch.cuda.synchronize()

        lines = []
        for start, end, meta in pairs:
            timing_source = "gpu"
            try:
                # elapsed_time returns milliseconds (float)
                ms = start.elapsed_time(end)
            except Exception:
                # Fallback to CPU-based timing; mark the record so downstream
                # analysis can treat it specially.
                ms = (time.time() - meta.get("cpu_ts", time.time())) * 1000.0
                timing_source = "cpu_fallback"

            record = {
                "rank": self.rank,
                "elapsed_ms": float(ms),
                "timing_source": timing_source,
                **meta,
            }
            lines.append(record)

        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except Exception:
                pass

        # Append as jsonlines (simple and robust for downstream merging).
        # Optionally use an advisory file lock to avoid interleaved writes
        # when multiple processes/threads attempt to write the same file.
        if use_file_lock:
            try:
                import fcntl

                with open(path, "a", encoding="utf-8") as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX)
                    except Exception:
                        # best-effort locking; continue if locking unavailable
                        pass
                    for rec in lines:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except Exception:
                        pass
            except Exception:
                # If fcntl/import fails, fall back to plain append
                with open(path, "a", encoding="utf-8") as f:
                    for rec in lines:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with open(path, "a", encoding="utf-8") as f:
                for rec in lines:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return len(lines)

    def aggregate_stats(self, records: List[Dict[str, Any]]):
        from collections import defaultdict
        stats = defaultdict(list)
        for r in records:
            key = (r.get("action"), r.get("step"), r.get("block"), r.get("layer"))
            stats[key].append(r["elapsed_ms"])
        out = {}
        import numpy as np
        for k, v in stats.items():
            arr = np.array(v)
            out[k] = {
                "count": len(v),
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }
        return out

from contextvars import ContextVar
CURRENT_CEM_STEP: ContextVar[int | None] = ContextVar("CURRENT_CEM_STEP", default=None)
# Context var for the current action chunk id (set by the inference driver per chunk)
CURRENT_ROLLOUT: ContextVar[int | None] = ContextVar("CURRENT_ROLLOUT", default=None)

""" hooks for profiling forward pass """
def _register_timing_hooks(model: torch.nn.Module, timer: CudaTimerCollection, name_filter=None) -> List:
    """
    Register pre/post forward hooks on modules whose name matches name_filter.
    name_filter: callable(name) -> bool or regex-like check
    Returns list of hook handles for later removal.
    """
    handles = []

    def make_hooks(mod_name: str):
        # pre-hook: record start event on appropriate device/stream
        def pre_hook(module, input):
            # determine device from first tensor input fallback to module params
            dev = None
            if isinstance(input, (list, tuple)) and len(input) and isinstance(input[0], torch.Tensor):
                dev = input[0].device
            else:
                try:
                    dev = next(module.parameters()).device
                except StopIteration:
                    dev = torch.device("cuda")
            stream = torch.cuda.current_stream(device=dev)
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            # put start and device on module to retrieve in post
            setattr(module, "_cuda_timer_start_pair", (start, dev))

        # post-hook: record end event and push pair + meta into timer's pending buffer
        def post_hook(module, input, output):
            pair = getattr(module, "_cuda_timer_start_pair", None)
            if pair is None:
                return
            start, dev = pair
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(device=dev))
            # assemble meta - include denoise step and current action chunk from contextvars
            cem_step = CURRENT_CEM_STEP.get(None)
            rollout = CURRENT_ROLLOUT.get(None)
            meta = {"cem_step": cem_step, "rollout": rollout, "layer_name": mod_name}
            # append to timer pending (use its lock or a method)
            with timer._lock:
                timer._pending_pairs.append((start, end, meta))

        return pre_hook, post_hook

    for name, sub in model.named_modules():
        if name_filter is None or (callable(name_filter) and name_filter(name)):
            pre, post = make_hooks(name)
            h1 = sub.register_forward_pre_hook(pre)
            h2 = sub.register_forward_hook(post)
            handles.extend([h1, h2])
    return handles

def unregister_hooks(handles: List):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass