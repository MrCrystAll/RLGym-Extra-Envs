from typing import Any, Dict, List

from rlgym.api import RewardFunction

from rlgym.lunar_lander.api.state import LunarLanderState


class LegHitReward(RewardFunction[str, LunarLanderState, float]):
    """
    Computes the per-step reward using potential-based shaping.

    Corresponds to RewardFunction in the RLGym API.
    """

    def __init__(self, reward: float = 10) -> None:
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

        _legs: list[int | float] = [
            leg.ground_contact * self.reward for leg in state.legs
        ]

        return {agent: sum(_legs) for agent in agents}
