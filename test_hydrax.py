"""Step 2: check that Hydrax runs a controller, on the JAX and Warp backends.

No viewer window (works on a headless machine like Codespaces).
For each backend, it builds the pendulum swing-up task, runs a few
predictive-sampling optimization steps from the hanging-down state,
and prints the lowest rollout cost found at each step.

Run:  uv run python test_hydrax.py          # both backends
      uv run python test_hydrax.py warp     # one backend
"""

import sys
import time

import jax

from hydrax.algs import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum


def run(impl: str, iterations: int = 5) -> None:
    print(f"\n=== backend: {impl} | JAX device: {jax.devices()[0].platform} ===")
    task = Pendulum(impl=impl)
    ctrl = PredictiveSampling(
        task,
        num_samples=32,       # 32 candidate control plans per step
        noise_level=0.1,
        plan_horizon=1.0,     # each plan looks 1 s ahead
        spline_type="zero",
        num_knots=11,
    )

    state = task.make_data()  # pendulum starts hanging straight down
    params = ctrl.init_params()
    optimize = jax.jit(ctrl.optimize)

    for i in range(iterations):
        t0 = time.time()
        params, rollouts = optimize(state, params)
        # total cost of each rollout = sum over its time steps
        best = float(rollouts.costs.sum(axis=-1).min())
        jax.block_until_ready(params)
        dt = time.time() - t0
        note = "  (includes one-time compile)" if i == 0 else ""
        print(f"step {i}: best rollout cost = {best:8.3f}   {dt:6.2f} s{note}")

    print(f"{impl} backend OK")


if __name__ == "__main__":
    backends = sys.argv[1:] or ["jax", "warp"]
    for b in backends:
        run(b)
