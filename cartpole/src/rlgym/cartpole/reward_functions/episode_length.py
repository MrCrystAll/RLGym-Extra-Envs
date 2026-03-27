from typing import Any, Dict, List

from rlgym.api import RewardFunction

from rlgym.cartpole.api import CartPoleState


class EpisodeLengthReward(RewardFunction[str, CartPoleState, float]):
    def __init__(self, sutton_barto_reward: bool) -> None:
        self._sutton_barto_reward = sutton_barto_reward

    def get_rewards(
        self,
        agents: List[str],
        state: CartPoleState,
        is_terminated: Dict[str, bool],
        is_truncated: Dict[str, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[str, float]:
        return {
            "cart": int(not is_terminated["cart"]) - 1
            if self._sutton_barto_reward
            else 1
        }

    def reset(
        self,
        agents: List[str],
        initial_state: CartPoleState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass
