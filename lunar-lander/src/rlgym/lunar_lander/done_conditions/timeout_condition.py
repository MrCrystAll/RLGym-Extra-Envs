from typing import Any, Dict, List

from rlgym.api import DoneCondition

from rlgym.lunar_lander.api.state import LunarLanderState


class TimeoutCondition(DoneCondition[str, LunarLanderState]):
    """The truncation condition used by the default CartPole environment

    See https://gymnasium.farama.org/environments/classic_control/cart_pole/#episode-end (3)
    """

    def __init__(self) -> None:
        self.max_time = 500
        self.ticks_until_truncation = 0

    def is_done(
        self, agents: List[str], state: LunarLanderState, shared_info: Dict[str, Any]
    ) -> Dict[str, bool]:
        print(self.ticks_until_truncation)
        self.ticks_until_truncation += 1
        return {agent: self.ticks_until_truncation > self.max_time for agent in agents}

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.ticks_until_truncation = 0
