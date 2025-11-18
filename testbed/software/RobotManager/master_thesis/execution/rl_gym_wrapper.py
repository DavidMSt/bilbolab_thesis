import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FrodoRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim):
        super().__init__()
        self.sim = sim

        # Example: continuous 2D action (vx, omega)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Example observation: (x, y, psi) of one agent
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -np.pi]),
            high=np.array([10, 10, np.pi]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()              # you implement this
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.sim.step(action)         # apply to your agent
        obs = self._get_obs()

        # reward example: -distance to goal
        dist = np.linalg.norm(obs[:2] - np.array([5, 5]))
        reward = -dist

        terminated = dist < 0.3  
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        ag = list(self.sim.agents.values())[0]
        return np.array([ag.state.x, ag.state.y, ag.state.psi], dtype=np.float32)