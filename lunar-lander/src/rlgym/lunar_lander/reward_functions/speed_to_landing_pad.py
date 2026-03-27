from typing import Any, Dict, List

import numpy as np

from rlgym.api import RewardFunction

from rlgym.lunar_lander.api.common_values import (
    LEG_DOWN,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rlgym.lunar_lander.api.state import LunarLanderState


class SpeedToLandingPadPunishment(RewardFunction[str, LunarLanderState, float]):
    """
    Computes the per-step reward using potential-based shaping.

    Corresponds to RewardFunction in the RLGym API.
    """

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

        vel = state.lander.linearVelocity

        vel_x = (vel.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        vel_y = (vel.y - (state.helipad_y + LEG_DOWN / SCALE)) / (
            VIEWPORT_H / SCALE / 2
        )

        return {
            agent: -float(np.linalg.norm(np.asarray([vel_x, vel_y])))
            for agent in agents
        }
