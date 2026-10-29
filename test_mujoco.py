"""Step 1: check that plain MuJoCo works (CPU only).

A ball starts 1 m above the floor and falls. If MuJoCo works,
its height drops from 1.0 and settles near 0.1 (the ball's radius).

Run:  uv run python test_mujoco.py
"""

import mujoco

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)  # build the world from the description
data = mujoco.MjData(model)                  # the world's current state

print(f"MuJoCo version: {mujoco.__version__}")
for i in range(1001):                        # 1000 steps x 0.002 s = 2 seconds
    if i % 100 == 0:
        print(f"t={data.time:.2f}s  height={data.qpos[2]:.3f}m")
    mujoco.mj_step(model, data)              # advance physics one step
