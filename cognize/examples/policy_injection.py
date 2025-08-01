from cognize.epistemic_state import EpistemicState
from cognize.policies import collapse_soft_decay, realign_tanh, threshold_adaptive

agent = EpistemicState()
agent.inject_policy(
    collapse=collapse_soft_decay,
    realign=realign_tanh,
    threshold=threshold_adaptive
)

for r in [0.1, 0.3, 0.7, 0.9]:
    agent.receive(r)

print(agent.log())
