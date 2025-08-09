import json
import numpy as np
from cognize import EpistemicState
from cognize.policies import collapse_soft_decay_fn, realign_tanh_fn, threshold_adaptive_fn

def to_py(x):
    if isinstance(x, (np.bool_,)):        return bool(x)
    if isinstance(x, (np.integer,)):      return int(x)
    if isinstance(x, (np.floating,)):     return float(x)
    if isinstance(x, (list, tuple)):      return [to_py(v) for v in x]
    if isinstance(x, dict):               return {k: to_py(v) for k, v in x.items()}
    return x

robot = EpistemicState(V0=0.5)
robot.inject_policy(
    collapse=collapse_soft_decay_fn,
    realign=realign_tanh_fn,
    threshold=threshold_adaptive_fn,
)

sensor_readings = [0.1, 0.3, 0.7, 0.9]
for reading in sensor_readings:
    robot.receive(reading)

print(json.dumps(to_py(robot.log()), indent=2, ensure_ascii=False))
