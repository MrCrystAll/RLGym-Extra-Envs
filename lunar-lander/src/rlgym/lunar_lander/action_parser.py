from typing import Any, Dict, List

import numpy as np

from rlgym.api import ActionParser

from rlgym.lunar_lander.state import LunarLanderState


class LunarLanderDiscreteActionParser(
    ActionParser[str, int, np.ndarray, LunarLanderState, tuple[str, int]]
):
    """
    Validates / clips the raw action and passes it through to the engine.

    0: do nothing

    1: fire left orientation engine

    2: fire main engine

    3: fire right orientation engine

    Corresponds to ActionParser in the RLGym API.
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
        actions: Dict[str, int],
        state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        _parsed_actions = {}

        for agent, action in actions.items():
            parsed = np.zeros((2,))
            match action:
                case 1 | 3:
                    parsed[1] = action - 2  # 1 -> -1, 3 -> 1 (Go left, GO right)
                case 2:
                    parsed[0] = 1.0  # Activate main engine

            _parsed_actions[agent] = parsed

        return _parsed_actions

    def get_action_space(self, agent: str) -> tuple[str, int]:
        return "discrete", 4
