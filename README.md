# RLHF on CartPole

Educational implementations of several preference- and feedback-based reinforcement-learning approaches applied to `CartPole-v1`.

This repository accompanies the included **RLHF Survey.pdf**, which discusses the methods and their tradeoffs in more detail.

## Implementations

- `base.py` — preference-based reward modeling and policy optimization in the style of early RLHF work, using simulated preference comparisons.
- `aihf.py` — AI-feedback experiment with a learned reward model and policy-gradient updates.
- `rrhf.py` — reward-ranking / feedback experiment using a learned reward model with a DQN-style policy.

These are compact research/learning implementations intended to make the algorithms easier to inspect, not production RL training infrastructure or canonical reference implementations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Each experiment is standalone:

```bash
python base.py
python aihf.py
python rrhf.py
```

Training can take a while depending on the experiment and hardware. The scripts may display learning curves and write model checkpoints when complete.
