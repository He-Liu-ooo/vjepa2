#!/usr/bin/env python3
"""Plot stacked bar summary from torch.cuda.Event jsonl profiling output.

Usage:
    python tools/plot_torch_cuda_event_breakdown.py

This script uses two hard-coded paths at the top of the file:
  - `jsonl_path`: path to the torch.cuda.Event jsonl file
  - `output_image_path`: output PNG; contains a timestamp in format
    `YYYY-MM-DD-HH-MM` appended to the base filename.

The script groups records by (cem_step, rollout). For each group it
collects per-block timings (block index is the integer before the first
dot in `layer_name`, e.g. "18.attn.qkv" -> block 18) and computes the
following components per-block:

- norm1, norm2: exact records for `*.norm1` and `*.norm2`
- mlp.fc1, mlp.fc2: exact records for `*.mlp.fc1` and `*.mlp.fc2`
- mlp.other = max( mlp - fc1 - fc2, 0 ) where `*.mlp` may exist
- attn.qkv = sum of all `*.attn.qkv` records for the block
- attn.proj = `*.attn.proj`
- attn.other = max( attn - attn.qkv - attn.proj, 0 ) where `*.attn` may exist

For each (cem_step, rollout) the script expects up to 24 blocks. It
averages each component across available blocks (only blocks present
are averaged). Then it draws a stacked bar per (cem_step, rollout)
with the 8 stacked components in the order requested.
"""

import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Hard-coded paths (no CLI). Update these two if you want a different file.
jsonl_path = 'output/profiling/torch_cuda_event/2025-12-09-13-36.jsonl'
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_image_path = f'output/profiling/torch_cuda_event/2025-12-09-13-36+{timestamp}.png'


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # ignore malformed lines
                continue
    return records


def parse_records(records):
    # data[(cem_step, rollout)][block] -> dict of layer sums
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for r in records:
        elapsed = float(r.get("elapsed_ms", 0.0))
        cem = r.get("cem_step", None)
        rollout = r.get("rollout", None)
        layer = r.get("layer_name", "")
        if cem is None or rollout is None or not layer:
            continue

        # parse block and rest
        if "." in layer:
            block_str, rest = layer.split(".", 1)
            try:
                block = int(block_str)
            except Exception:
                # skip non-numeric block names
                continue
        else:
            # if no dot, skip
            continue

        key = (int(cem), int(rollout))
        blk = data[key][block]

        # classify
        if rest == "norm1":
            blk["norm1"] += elapsed
        elif rest == "norm2":
            blk["norm2"] += elapsed
        elif rest == "mlp.fc1":
            blk["mlp.fc1"] += elapsed
        elif rest == "mlp.fc2":
            blk["mlp.fc2"] += elapsed
        elif rest == "mlp":
            blk["mlp"] += elapsed
        elif rest == "attn.qkv":
            blk["attn.qkv"] += elapsed
        elif rest == "attn.proj":
            blk["attn.proj"] += elapsed
        elif rest == "attn":
            blk["attn"] += elapsed
        else:
            # collect other mlp/attn pieces under generic keys if helpful
            # but we won't use them directly
            blk.setdefault("other", 0.0)
            blk["other"] += elapsed

    return data


def aggregate_group(blkdicts):
    """Given blkdicts: mapping block -> layer dict, compute averaged components.

    Returns dict with keys: norm1,norm2,mlp.fc1,mlp.fc2,mlp.other,attn.qkv,attn.proj,attn.other
    """
    blocks = sorted(blkdicts.keys())
    comps = [
        "norm1",
        "norm2",
        "mlp.fc1",
        "mlp.fc2",
        "mlp.other",
        "attn.qkv",
        "attn.proj",
        "attn.other",
    ]

    per_block_vals = {c: [] for c in comps}

    for b in blocks:
        d = blkdicts[b]
        norm1 = d.get("norm1", 0.0)
        norm2 = d.get("norm2", 0.0)
        mlp_fc1 = d.get("mlp.fc1", 0.0)
        mlp_fc2 = d.get("mlp.fc2", 0.0)
        mlp_total = d.get("mlp", 0.0)
        attn_qkv = d.get("attn.qkv", 0.0)
        attn_proj = d.get("attn.proj", 0.0)
        attn_total = d.get("attn", 0.0)

        mlp_other = max(mlp_total - mlp_fc1 - mlp_fc2, 0.0)
        attn_other = max(attn_total - attn_qkv - attn_proj, 0.0)

        per_block_vals["norm1"].append(norm1)
        per_block_vals["norm2"].append(norm2)
        per_block_vals["mlp.fc1"].append(mlp_fc1)
        per_block_vals["mlp.fc2"].append(mlp_fc2)
        per_block_vals["mlp.other"].append(mlp_other)
        per_block_vals["attn.qkv"].append(attn_qkv)
        per_block_vals["attn.proj"].append(attn_proj)
        per_block_vals["attn.other"].append(attn_other)

    # average across blocks (use np.mean; if no blocks, zeros)
    out = {}
    for c in comps:
        vals = per_block_vals[c]
        out[c] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return out


def plot_summary(data, outpath=None, show=False):
    # data: mapping (cem,rollout) -> block -> layer dict
    groups = sorted(data.keys())
    if not groups:
        raise SystemExit("no data found to plot")

    labels = [f"({g[0]}, {g[1]})" for g in groups]
    agg_vectors = []
    for g in groups:
        agg = aggregate_group(data[g])
        # ordering
        vec = [
            agg["norm1"],
            agg["norm2"],
            agg["mlp.fc1"],
            agg["mlp.fc2"],
            agg["mlp.other"],
            agg["attn.qkv"],
            agg["attn.proj"],
            agg["attn.other"],
        ]
        agg_vectors.append(vec)

    arr = np.array(agg_vectors)  # shape (G, 8)

    comps = [
        "norm1",
        "norm2",
        "mlp.fc1",
        "mlp.fc2",
        "mlp.other",
        "attn.qkv",
        "attn.proj",
        "attn.other",
    ]

    # color palette: 2 greys for norm1/norm2, 3 greens for MLP parts,
    # and 3 blues for attention parts (order matches `comps` above)
    colors = [
        "#4b4a4a",  # norm1 (dark grey)
        "#a6a4a4",  # norm2 (light grey)
        "#195619",  # mlp.fc1 (green)
        "#3fbf4f",  # mlp.fc2 (green)
        "#98df8a",  # mlp.other (light green)
        "#104e79",  # attn.qkv (blue)
        "#2874B6",  # attn.proj (blue)
        "#77ACE8",  # attn.other (light blue)
    ]

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 6))

    bottom = np.zeros(len(groups))
    for i, comp in enumerate(comps):
        vals = arr[:, i]
        ax.bar(x, vals, bottom=bottom, color=colors[i], label=comp)
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=18)
    ax.set_ylabel("ms (avg per-block)", fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_title("Profiling summary: avg component time per (cem_step, rollout)", fontsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=18)
    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Saved summary to {outpath}")
    if show:
        plt.show()


def main():
    # Use the hard-coded `jsonl_path` and `output_image_path` defined at the
    # top of this script. This avoids CLI parsing as requested.
    if not os.path.exists(jsonl_path):
        raise SystemExit(f"jsonl not found: {jsonl_path}")

    records = load_jsonl(jsonl_path)
    data = parse_records(records)
    plot_summary(data, outpath=output_image_path, show=False)


if __name__ == "__main__":
    main()
