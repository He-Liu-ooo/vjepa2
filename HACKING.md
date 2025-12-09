# HACKING

This document collects quick developer tips for running the demo with
verbose logging and for enabling profiling (torch.profiler and
torch.cuda.Event timing). Use these instructions when you are
investigating performance or debugging the MPC / world-model path.

## Enable verbose logging

Edit `notebooks/energy_landscape_example.py` and update the
`LOGGING` configuration so the root logger is set to `DEBUG`. For
example:

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {'format': '[%(levelname)s] %(asctime)s %(name)s %(message)s'}
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'DEBUG',
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    }
}
```

Note: some modules still call `logging.basicConfig()` at import time
and may interfere with this configuration. If you still see missing
DEBUG messages, either remove those module-level calls or ensure the
entrypoint config runs early and clears existing root handlers.

## Profiling

Two profiling options are supported in the codebase:

- `torch.profiler` (detailed, heavy traces)
- `torch.cuda.Event` via the project's `CudaTimerCollection` (lighter,
  per-module timing written as JSONL)

Both are controlled by flags in `notebooks/utils/mpc_utils.py`'s
`cem()` function.

### torch.profiler

To enable `torch.profiler`, set the `enable_torch_profiler` flag to
`True` in the `cem()` signature and pass a `torch_profiler_dir` where
the traces will be written. Example signature excerpt:

```python
def cem(...,
        enable_torch_profiler=False,
        torch_profiler_dir="/path/to/vjepa2/output/profiling/torch_profiler",
        ...):
    ...
```

After running the instrumented demo you can visualize traces with
TensorBoard. From the repository root run:

```bash
./tensorboard.sh
```

Adjust the `cem_steps` / `rollout` parameters in
`notebooks/energy_landscape_example.py` to control where and how many
steps are profiled.

### torch.cuda.Event (CudaTimerCollection)

The codebase includes a `CudaTimerCollection` and hooks that attach
to module forwards to emit lightweight timing records to a JSONL file.
These are useful for quick per-layer breakdowns without the overhead
of `torch.profiler`.

Files involved:

- `vjepa2-zipball-main-local/.../src/utils/profiling.py` (project
  helper)
- `notebooks/utils/mpc_utils.py` (registers hooks and controls output
  path)

Enable it by setting `enable_torch_cuda_event=True` in the `cem()`
function and providing a `torch_cuda_event_dir`:

```python
def cem(...,
        enable_torch_cuda_event=True,
        torch_cuda_event_dir="/path/to/vjepa2/output/profiling/torch_cuda_event",
        ...):
    ...
```

The produced JSONL file can be summarized with the included plotting
tool:

```bash
python tools/plot_torch_cuda_event_breakdown.py
```

That script expects the JSONL file path to be set at the top of the
script (it currently uses a hard-coded `jsonl_path` and writes an
output PNG with a `YYYY-MM-DD-HH-MM` timestamp appended).