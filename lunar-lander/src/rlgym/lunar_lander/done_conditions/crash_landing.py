from typing import Any, Dict, List

from rlgym.api import DoneCondition

from rlgym.lunar_lander.api.common_values import SCALE, VIEWPORT_W
from rlgym.lunar_lander.api.state import LunarLanderState


class CrashLandingCondition(DoneCondition[str, LunarLanderState]):
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
            return {agent: True for agent in agents}
        if not state.lander.awake:
            return {agent: True for agent in agents}
        return {agent: False for agent in agents}
