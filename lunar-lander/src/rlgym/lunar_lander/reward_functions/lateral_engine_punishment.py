from typing import Any, Dict, List

from rlgym.api import RewardFunction

from rlgym.lunar_lander.api.state import LunarLanderState


class LateralEnginePunishment(RewardFunction[str, LunarLanderState, float]):
    """
    Computes the per-step reward using potential-based shaping.

    Corresponds to RewardFunction in the RLGym API.
    """

    def __init__(self, punishment: float = -0.03) -> None:
        self.punishment = punishment

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

        return {agent: self.punishment for agent in agents}
