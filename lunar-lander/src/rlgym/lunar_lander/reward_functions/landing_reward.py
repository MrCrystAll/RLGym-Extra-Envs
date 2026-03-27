from typing import Any, Dict, List

from rlgym.api import RewardFunction

from rlgym.lunar_lander.api.common_values import (
    SCALE,
    VIEWPORT_W,
)
from rlgym.lunar_lander.api.state import LunarLanderState


class LandingReward(RewardFunction[str, LunarLanderState, float]):
    """
    Computes the per-step reward using potential-based shaping.

    Corresponds to RewardFunction in the RLGym API.
    """

    def __init__(self, reward: float = 100) -> None:
        self.reward = reward

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[str],
        state: LunarLanderState,
        is_terminated: Dict[str, bool],
        is_truncated: Dict[str, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {}

        pos_x = (state.lander.position.x - VIEWPORT_W / SCALE / 2) / (
            VIEWPORT_W / SCALE / 2
        )

        for agent in agents:
            if state.game_over or abs(pos_x) >= 1.0:
                rewards[agent] = -self.reward
            elif not state.lander.awake:
                rewards[agent] = self.reward

        return rewards
