from typing import Any, Dict, List

from rlgym.api import DoneCondition

from rlgym.lunar_lander.common_values import AGENT_NAME, SCALE, VIEWPORT_W
from rlgym.lunar_lander.state import LunarLanderState


class LunarLanderTermination(DoneCondition[str, LunarLanderState]):
    """
    Returns True when the episode should end (crash or safe landing).

    Corresponds to DoneCondition (termination) in the RLGym API.
    """

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def is_done(
        self, agents: List[str], state: LunarLanderState, shared_info: Dict[str, Any]
    ) -> Dict[str, bool]:
        if state.game_over or abs(state.lander.position.x / VIEWPORT_W * SCALE) >= 1.0:
            return {AGENT_NAME: True}
        if not state.lander.awake:
            return {AGENT_NAME: True}
        return {AGENT_NAME: False}


class LunarLanderTruncation(DoneCondition[str, LunarLanderState()]):
    def __init__(self) -> None:
        self.max_time = 500
        self.ticks_until_truncation = 0

    def is_done(
        self, agents: List[str], state: LunarLanderState, shared_info: Dict[str, Any]
    ) -> Dict[str, bool]:
        self.ticks_until_truncation += 1
        return {AGENT_NAME: self.ticks_until_truncation > self.max_time}

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.ticks_until_truncation = 0
