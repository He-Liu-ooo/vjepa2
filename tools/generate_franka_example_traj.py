import numpy as np

# Small helper to create a synthetic franka_example_traj.npz used by the demo
# Creates:
# - observations: shape (1, T, H, W, C) uint8 in [0,255]
# - states: shape (1, T, 7) floats representing poses

T = 8
H = 256
W = 256
C = 3

observations = (np.random.rand(1, T, H, W, C) * 255).astype(np.uint8)
# states: e.g., 7-d pose vector (x,y,z,qx,qy,qz,qw) arbitrary
states = np.random.randn(1, T, 7).astype(np.float32)

np.savez_compressed("franka_example_traj.npz", observations=observations, states=states)
print("Wrote franka_example_traj.npz (synthetic): observations", observations.shape, "states", states.shape)
