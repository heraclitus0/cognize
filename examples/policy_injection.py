from cognize import EpistemicState
from cognize.policies import (
    collapse_soft_decay_fn,
    realign_tanh_fn,
    threshold_adaptive_fn
)

agent = EpistemicState()
agent.inject_policy(
    collapse=collapse_soft_decay_fn,
    realign=realign_tanh_fn,
    threshold=threshold_adaptive_fn
)

for r in [0.1, 0.3, 0.7, 0.9]:
    agent.receive(r)

print(agent.log())
