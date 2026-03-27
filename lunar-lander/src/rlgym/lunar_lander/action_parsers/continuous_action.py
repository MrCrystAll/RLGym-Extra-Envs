from typing import Any, Dict, List

import numpy as np

from rlgym.api import ActionParser

from rlgym.lunar_lander.api.state import LunarLanderState


class ContinuousAction(
    ActionParser[str, np.ndarray, np.ndarray, LunarLanderState, tuple[str, int]]
):
    """
    Validates / clips the raw action and passes it through to the engine.
    """

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def parse_actions(
        self,
        actions: Dict[str, np.ndarray],
        state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        _clipped_actions = {}

        for agent, action in actions.items():
            assert action.size == 2, (
                f"Expected 2 bins in the actions but got {action.size}"
            )

            _clipped_actions[agent] = np.clip(action, -1, 1)

        return _clipped_actions

    def get_action_space(self, agent: str) -> tuple[str, int]:
        return "continuous", 2
