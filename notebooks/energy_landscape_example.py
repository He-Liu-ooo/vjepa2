import sys
import os
# Ensure the repository root (one level up from this notebook/script) is on sys.path.
import logging
import logging.config
# Using __file__ makes the import robust no matter the current working directory.
HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import logging
import logging.config

# --- remove any handlers previously attached to root ---
root = logging.getLogger()
for h in list(root.handlers):   # copy list() to avoid mutation-during-iteration
    root.removeHandler(h)

# --- now configure logging centrally ---
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
        'level': 'INFO',
    }
}
logging.config.dictConfig(LOGGING)
# Define a module logger after centralized configuration so subsequent
# `logger.*` calls work as expected (prevents NameError when using `logger`).
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import inspect
import importlib

from app.vjepa_droid.transforms import make_transforms
from utils.mpc_utils import (
    compute_new_pose,
    poses_to_diff,
)

# Initialize VJEPA 2-AC model
repo_dir = "/home/hel19/workspace/repos/neural_network/vjepa2/vjepa2-zipball-main-local/facebookresearch-vjepa2-c2963a4"
# encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
encoder, predictor = torch.hub.load(repo_dir, "vjepa2_ac_vit_giant", source="local")

# Move models to device (GPU if available) so subsequent calls run on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
predictor = predictor.to(device)
logger.info("Using device: %s", device)

# Print/log the encoder class so we can verify what implementation was loaded.
logger.info("Loaded encoder type: %s", encoder.__class__.__module__ + "." + encoder.__class__.__name__)
logger.info("Loaded predictor type: %s", predictor.__class__.__module__ + "." + predictor.__class__.__name__)

# Initialize transform
crop_size = 256
tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0.,
    auto_augment=False,
    motion_shift=False,
    crop_size=crop_size,
)

def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    # ensure input is on the same device as the models
    c = c.to(device)
    logger.debug("Before permute: c: %s", c.shape)
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    logger.debug("Before encoder: c: %s", c.shape)
    h = encoder(c)
    logger.debug("After encoder: h: %s", h.shape)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    logger.debug("After view: h: %s", h.shape)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
        logger.debug("After layer_norm: h: %s", h.shape)
    return h


def forward_actions(z, nsamples, grid_size=0.075, normalize_reps=True, action_repeat=1):

    def make_action_grid(grid_size=grid_size):
        action_samples = []
        for da in np.linspace(-grid_size, grid_size, nsamples):
            for db in np.linspace(-grid_size, grid_size, nsamples):
                for dc in np.linspace(-grid_size, grid_size, nsamples):
                    action_samples += [torch.tensor([da, db, dc, 0, 0, 0, 0], device=z.device, dtype=z.dtype)]
        return torch.stack(action_samples, dim=0).unsqueeze(1)

    # Sample grid of actions
    action_samples = make_action_grid()
    logger.info("Sampled grid of actions; num actions = %d", len(action_samples))

    def step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
        return _z, _s

    # Context frame rep and context pose
    z_hat = z[:, :tokens_per_frame].repeat(int(nsamples**3), 1, 1)  # [S, N, D]
    s_hat = states[:, :1].repeat((int(nsamples**3), 1, 1))  # [S, 1, 7]
    a_hat = action_samples  # [S, 1, 7]

    for _ in range(action_repeat):
        _z, _s = step_predictor(z_hat, a_hat, s_hat)
        z_hat = torch.cat([z_hat, _z], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    return z_hat, s_hat, a_hat


def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, N, D]
    loss = torch.mean(loss, dim=[1, 2])
    return loss.tolist()


def main():
    # Load robot trajectory
    play_in_reverse = False  # Use this FLAG to try loading the trajectory backwards, and see how the energy landscape changes

    # print("1")
    trajectory = np.load("input/franka_example_traj.npz")
    # print("2")
    np_clips = trajectory["observations"]
    np_states = trajectory["states"]
    if play_in_reverse:
        np_clips = trajectory["observations"][:, ::-1].copy()
        np_states = trajectory["states"][:, ::-1].copy()
    # [1, 1, D_act], derived from the difference of states
    np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], np_states[0, 1]), axis=(0, 1))

    # Convert trajectory to torch tensors
    global clips, states, actions
    # move input tensors to the device used for models
    clips = transform(np_clips[0]).unsqueeze(0).to(device)
    states = torch.tensor(np_states).to(device)
    actions = torch.tensor(np_actions).to(device)
    logger.info("clips: %s; states: %s; actions: %s", clips.shape, states.shape, actions.shape)

    # Visualize loaded video frames from traj
    T = len(np_clips[0])
    plt.figure(figsize=(20, 3))
    _ = plt.imshow(np.transpose(np_clips[0], (1, 0, 2, 3)).reshape(256, 256 * T, 3))

    # Compute energy for cartesian action grid of size (nsample x nsamples x nsamples)
    nsamples = 5
    grid_size = 0.075
    with torch.no_grad():
        h = forward_target(clips)
        z_hat, s_hat, a_hat = forward_actions(h, nsamples=nsamples, grid_size=grid_size)
        loss = loss_fn(z_hat, h)  # jepa prediction loss

    # Plot the energy
    plot_data = []
    for b, v in enumerate(loss):
        # a_hat elements may be tensors on GPU — convert to Python floats before using numpy
        dx = a_hat[b, :-1, 0].sum().cpu().item() if hasattr(a_hat[b, :-1, 0], 'cpu') else float(a_hat[b, :-1, 0].sum())
        dy = a_hat[b, :-1, 1].sum().cpu().item() if hasattr(a_hat[b, :-1, 1], 'cpu') else float(a_hat[b, :-1, 1].sum())
        dz = a_hat[b, :-1, 2].sum().cpu().item() if hasattr(a_hat[b, :-1, 2], 'cpu') else float(a_hat[b, :-1, 2].sum())
        plot_data.append((dx, dy, dz, float(v)))

    delta_x = [d[0] for d in plot_data]
    delta_y = [d[1] for d in plot_data]
    delta_z = [d[2] for d in plot_data]
    energy = [d[3] for d in plot_data]

    # Ground-truth actions may be tensors on device; convert to Python floats for logging
    gt_x = actions[0, 0, 0].cpu().item() if hasattr(actions[0, 0, 0], 'cpu') else float(actions[0, 0, 0])
    gt_y = actions[0, 0, 1].cpu().item() if hasattr(actions[0, 0, 1], 'cpu') else float(actions[0, 0, 1])
    gt_z = actions[0, 0, 2].cpu().item() if hasattr(actions[0, 0, 2], 'cpu') else float(actions[0, 0, 2])

    # Create the 2D histogram
    heatmap, xedges, yedges = np.histogram2d(delta_x, delta_z, weights=energy, bins=nsamples)

    # Set axis labels
    plt.xlabel("Action Delta x")
    plt.ylabel("Action Delta z")
    plt.title(f"Energy Landscape")

    # Display the heatmap
    logger.info("Ground truth action (x,y,z) = (%.2f,%.2f,%.2f)", gt_x, gt_y, gt_z)
    _ = plt.imshow(heatmap.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis")
    _ = plt.colorbar()
    plt.show()

    # Compute the optimal action using MPC
    from utils.world_model_wrapper import WorldModel

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        # Doing very few CEM iterations with very few samples just to run efficiently on CPU...
        # ... increase cem_steps and samples for more accurate optimization of energy landscape
        mpc_args={
            "rollout": 4,
            "samples": 25,
            "topk": 10,
            "cem_steps": 4,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.075,
            "verbose": True
        },
        normalize_reps=True,
        device=device,
    )

    # root = logging.getLogger()
    # print('root.level:', root.level, 'root.getEffectiveLevel():', root.getEffectiveLevel())
    # print('root.handlers:', root.handlers)
    # lg = logging.getLogger(__name__)
    # print('__name__ logger level:', lg.level, 'effective:', lg.getEffectiveLevel())
    # print('isEnabledFor DEBUG:', lg.isEnabledFor(logging.DEBUG))
    # for i,h in enumerate(root.handlers):
    #     print(i, type(h), 'handler.level =', h.level)
    
    with torch.no_grad():
        h = forward_target(clips)
        z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
        s_n = states[:, :1]
        logger.info("Starting planning using Cross-Entropy Method...")

        # Debug: print shapes of the tensors used for planning
        try:
            logger.debug("z_n shape: %s", tuple(z_n.shape))
            logger.debug("s_n shape: %s", tuple(s_n.shape))
            logger.debug("z_goal shape: %s", tuple(z_goal.shape))
            logger.debug("tokens_per_frame: %d", tokens_per_frame)
        except Exception:
            logger.debug("Could not log tensor shapes for z_n/s_n/z_goal")

        actions_out = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()

    logger.info(
        "Actions returned by planning with CEM (x,y,z) = (%.2f,%.2f,%.2f)",
        actions_out[0, 0],
        actions_out[0, 1],
        actions_out[0, 2],
    )


if __name__ == "__main__":
    main()
