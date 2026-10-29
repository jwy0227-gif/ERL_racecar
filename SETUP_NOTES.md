# MuJoCo + Hydrax setup notes

## Status
- [x] MuJoCo installed and tested (`test_mujoco.py`, CPU)
- [x] Hydrax installed and running a controller on the JAX backend (`test_hydrax.py jax`, CPU)
- [ ] Hydrax Warp backend (`impl="warp"`): **needs an NVIDIA GPU**, cannot run on CPU (see below)

## How to run
```bash
pip install uv                      # once
uv sync                             # installs everything (~5 GB, includes NVIDIA CUDA libraries)
uv run python test_mujoco.py        # plain MuJoCo
uv run python test_hydrax.py jax    # Hydrax, JAX backend (works on CPU)
uv run python test_hydrax.py warp   # Hydrax, Warp backend (needs NVIDIA GPU)
```

## Results (cloud Linux machine, CPU only, no GPU)

### test_mujoco.py
Ball height drops from 1.000 m and settles at 0.100 m (its radius).
At t = 0.2 s the simulator gives 0.802 m vs. free-fall theory 1 - 0.5*9.81*0.2^2 = 0.804 m.

### test_hydrax.py jax
```
=== backend: jax | JAX device: cpu ===
step 0: best rollout cost =    7.980     2.94 s  (includes one-time compile)
step 1: best rollout cost =    7.958     1.83 s
step 2: best rollout cost =    7.920     0.00 s
step 3: best rollout cost =    7.879     0.00 s
step 4: best rollout cost =    7.838     0.00 s
jax backend OK
```
The best cost goes down every step, so the controller is improving its plan.

A long red `Unable to load cuPTI` / `Jax plugin configuration error` traceback prints first.
That is JAX looking for an NVIDIA GPU, not finding one, and falling back to CPU. It is harmless here.

### test_hydrax.py warp
```
RuntimeError: Unknown backend cuda. Available backends are ['cpu']
```
Cause: in `mujoco/mjx/_src/io.py`, the Warp implementation explicitly requires a CUDA GPU
(`# WARP implementation requires a CUDA GPU.`). So, through Hydrax/MJX, Warp has no CPU fallback.
(Standalone `mujoco_warp` documents CPU support for debugging, but the MJX path Hydrax uses does not allow it.)

## Issues found and fixed
1. **Version conflict.** With the latest `mujoco` (3.17.0), importing Hydrax fails:
   `TypeError: kernel() got an unexpected keyword argument 'grid_stride'`.
   MuJoCo 3.17's built-in Warp code needs a newer `warp-lang` than Hydrax allows (`<1.13.0`).
   Fix: pin `mujoco==3.8.0` and `mujoco-mjx==3.8.0`, the versions in Hydrax's own `uv.lock`.

## Next step
Run `uv run python test_hydrax.py warp` on a machine with an NVIDIA GPU
(lab workstation, campus GPU cluster, or a cloud GPU) to finish the Warp part.
