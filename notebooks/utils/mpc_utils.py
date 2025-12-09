# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time
import os, sys
import contextlib
from datetime import datetime
import importlib.util

from src.utils.logging import get_logger

# Use get_logger without forcing a reconfiguration of the root logger here.
# Calling with `force=True` would call `logging.basicConfig(..., force=True)`
# and override any centralized `dictConfig` the entrypoint applied earlier.
logger = get_logger(__name__)

def _load_profiling_from_zipball():
    # compute repo root relative to this file
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    zip_src = os.path.join(repo_root, "vjepa2-zipball-main-local", "facebookresearch-vjepa2-c2963a4", "src")
    profiling_path = os.path.join(zip_src, "utils", "profiling.py")
    if not os.path.exists(profiling_path):
        return None
    spec = importlib.util.spec_from_file_location("zip_profiling", profiling_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_zip_mod = _load_profiling_from_zipball()
if _zip_mod is not None:
    CudaTimerCollection = getattr(_zip_mod, "CudaTimerCollection", None)
    _register_timing_hooks = getattr(_zip_mod, "_register_timing_hooks", None)
    unregister_hooks = getattr(_zip_mod, "unregister_hooks", lambda handles: None)
    CURRENT_CEM_STEP = getattr(_zip_mod, "CURRENT_CEM_STEP", None)
    CURRENT_ROLLOUT = getattr(_zip_mod, "CURRENT_ROLLOUT", None)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    world_model_module,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
    enable_torch_profiler=False,
    torch_profiler_dir="/home/hel19/workspace/repos/neural_network/vjepa2/output/profiling/torch_profiler",
    enable_torch_cuda_event=True,
    torch_cuda_event_dir="/home/hel19/workspace/repos/neural_network/vjepa2/output/profiling/torch_cuda_event",
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    # ensure torch_profiler_dir is a directory path; append timestamp to keep traces per-run
    # timestamp format: YYYY-MM-DD-HH-MM
    # Note: torch_profiler_dir parameter may be a base path; we create a subfolder with timestamp
    # to avoid overwriting previous profiling runs.
    
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    # sometimes you want to fix certain axes of the action to narrow down the search space
    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):
            
            try:
                dev = mean.device
                if getattr(dev, 'type', None) == 'cuda':
                    torch.cuda.synchronize()
            except Exception:
                dev = None
            t0 = time.time()
            
            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
            action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    torch.zeros((len(action_samples), 3), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None]
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            try:
                logger.debug(
                    "Before world_model: frame_traj: %s, action_traj: %s, pose_traj: %s",
                    tuple(frame_traj.shape),
                    tuple(action_traj.shape),
                    tuple(pose_traj.shape),
                )
            except Exception:
                logger.debug("Could not log WM input shapes")

            token = CURRENT_ROLLOUT.set(h)
            try:
                next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            finally:
                CURRENT_ROLLOUT.reset(token)
            
            try:
                logger.debug(
                    "After world_model: next_frame: %s, next_pose: %s",
                    tuple(next_frame.shape),
                    tuple(next_pose.shape),
                )
            except Exception:
                logger.debug("Could not log WM output shapes")
                
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)
            
            # finalize timing for this rollout step
            try:
                if dev is not None and getattr(dev, 'type', None) == 'cuda':
                    torch.cuda.synchronize()
            except Exception:
                pass
            t1 = time.time()
            elapsed_ms = (t1 - t0) * 1000.0
            logger.debug(f"rollout step {h} timing: {elapsed_ms:.3f} ms")

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    # Configure profiler context when requested. When profiling is disabled we
    # use a nullcontext so the loop body stays the same but no profiler is
    # instantiated (and no profile.step() is called).
    # logger.info(f"cem_steps: {cem_steps}, rollout: {rollout}, samples: {samples}, topk: {topk}")
    if enable_torch_profiler:
        # append timestamped subfolder to torch_profiler_dir
        logger.info("Using torch.profiler for detailed timing measurements during inference.")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        torch_profiler_dir = os.path.join(torch_profiler_dir, timestamp)
        os.makedirs(torch_profiler_dir, exist_ok=True)
        prof_cm = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profiler_dir, worker_name="worker0"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )
    else:
        prof_cm = contextlib.nullcontext()
    
    if enable_torch_cuda_event:
        logger.info("Using torch.cuda.Event for detailed timing measurements during inference.")
        torch_cuda_event_timer = CudaTimerCollection(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
        # Register hooks on the concrete network module used by the inference model.
        # Prefer `model.net` if present, otherwise register on the model itself.
        net_module = getattr(world_model_module, "predictor_blocks", world_model_module)
        handles = _register_timing_hooks(net_module, torch_cuda_event_timer, name_filter=lambda n: ("mlp" in n) or ("attn" in n) or ("norm1" in n) or ("norm2" in n))
    else:
        torch_cuda_event_timer = None
        handles = None
        
    try:
        with prof_cm as prof:
            for step in tqdm(range(cem_steps), disable=True):
                token = CURRENT_CEM_STEP.set(step)
                try:
                    action_traj, frame_traj = sample_action_traj()
                finally:
                    CURRENT_CEM_STEP.reset(token)
                    
                selected_actions = select_topk_action_traj(
                    final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
                )
                mean_selected_actions = selected_actions.mean(dim=0)
                std_selected_actions = selected_actions.std(dim=0)

                # -- Update new sampling mean and std based on the top-k samples
                mean = torch.cat(
                    [
                        mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                        mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                        + mean[..., -1:] * momentum_mean_gripper,
                    ],
                    dim=-1,
                )
                std = torch.cat(
                    [
                        std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                        std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
                    ],
                    dim=-1,
                )

                logger.debug(f"new mean: {mean.sum(dim=0)} {std.sum(dim=0)}")

                if enable_torch_profiler:
                    try:
                        prof.step()
                    except Exception:
                        pass
    finally:
        # Ensure hooks are unregistered and any pending timing records are flushed.
        if torch_cuda_event_timer is not None:
            try:
                if handles is not None:
                    unregister_hooks(handles)
            except Exception:
                logger.exception("Error while unregistering timing hooks")
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
                outpath = f"{torch_cuda_event_dir}/{timestamp}.jsonl"
                torch_cuda_event_timer.flush_to_file(outpath, batch_sync=True, use_file_lock=True)
            except Exception:
                logger.exception("Error while flushing torch_cuda_event_timer output file")

    new_action = torch.cat(
        [
            mean[..., :3],
            torch.zeros((rollout, 3), device=mean.device),
            round_small_elements(mean[..., -1:], 0.25),
        ],
        dim=-1,
    )[None, :]

    return new_action


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.numpy()
        end = end.numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)
