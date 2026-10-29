# MuJoCo + Hydrax setup notes

## Status
- [x] MuJoCo installed and tested (`test_mujoco.py`, CPU only)
- [~] Hydrax added as a dependency. Install finished, but import not yet verified
- [ ] Hydrax Warp backend (`Pendulum(impl="warp")`) not yet tested

## How to run
```bash
pip install uv      # once
uv sync             # installs everything (~5 GB, includes NVIDIA CUDA libraries)
uv run python test_mujoco.py
```

Expected output: ball height drops from 1.000 m and settles at 0.100 m (its radius).
At t = 0.2 s the simulator gives 0.802 m vs. free-fall theory 1 - 0.5*9.81*0.2^2 = 0.804 m.

## Issue found
With the latest `mujoco` (3.17.0), importing Hydrax fails:
```
TypeError: kernel() got an unexpected keyword argument 'grid_stride'
```
Cause: MuJoCo 3.17's built-in Warp code needs a newer `warp-lang` than Hydrax allows
(Hydrax pins `warp-lang>=1.12.1,<1.13.0`).
Fix applied: pin `mujoco==3.8.0` and `mujoco-mjx==3.8.0`, the versions in Hydrax's own `uv.lock`.
Not yet confirmed that the import works after this pin.

## Next step
```bash
uv run python -c "from hydrax.tasks.pendulum import Pendulum; Pendulum(); print('OK')"
uv run python -c "from hydrax.tasks.pendulum import Pendulum; Pendulum(impl='warp'); print('Warp OK')"
```
