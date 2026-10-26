# SARSA on CliffWalking-v1

A tabular implementation of the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm, applied to the CliffWalking-v1 environment from [Gymnasium](https://gymnasium.farama.org/).

## Environment

CliffWalking is a 4×12 grid world. The agent starts in the bottom-left corner and must reach the bottom-right goal. The bottom row between start and goal is a cliff: stepping into it gives a reward of -100 and sends the agent back to the start. Every other step costs -1.

- States: 48 (one per grid cell)
- Actions: 4 (up, right, down, left)
- Optimal return: -13 (walks along the cliff edge)

## Algorithm

SARSA is an on-policy TD control method. It updates the Q-table using the action the agent actually takes next:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \, Q(s', a') - Q(s, a) \big]$$

Because the next action a' is chosen by the same epsilon-greedy policy used for behavior, SARSA accounts for its own exploration. This makes it more cautious near the cliff than Q-learning.

## Hyperparameters

| Parameter  | Value | Description                       |
|------------|-------|-----------------------------------|
| gamma    | 0.99  | Discount factor                   |
| alpha    | 0.5   | Learning rate                     |
| epsilon  | 0.1   | Exploration rate (epsilon-greedy) |
| episodes | 500   | Training episodes                 |

## Installation

pip install "gymnasium[toy-text]" numpy## Usage

1. Open SARSA.ipynb in Jupyter.
2. Run all cells from top to bottom.
3. Every 50 episodes, the environment renders in human mode so you can watch the agent's progress.

A final greedy run (argmax over the learned Q-table) prints the total reward and episode length of the learned policy.

## Results

| Metric                         | Value           |
|--------------------------------|-----------------|
| Final training episode reward  | -24 (length 24) |
| Greedy policy evaluation       | -17 (length 17) |

Early episodes show large negative returns (e.g., -1384) because the agent falls off the cliff often while exploring. Returns improve steadily as the Q-table converges. The learned SARSA policy takes a safer route, a few steps away from the cliff, rather than the optimal edge path.

## Project Structure

.
├── SARSA.ipynb   # SARSA implementation and experiments
└── README.md## References

- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd ed.), Section 6.4 (SARSA).
- [Gymnasium CliffWalking documentation](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
