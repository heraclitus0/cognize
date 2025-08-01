from cognize.epistemic_state import EpistemicState

agent = EpistemicState()

for r in [0.1, 0.3, 0.6, 0.4]:
    agent.receive(r)

print(agent.summary())
