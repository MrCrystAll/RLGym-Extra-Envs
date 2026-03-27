from typing import Any, Dict, List

import numpy as np

from rlgym.api import ObsBuilder

from rlgym.lunar_lander.api.common_values import (
    TICKS_PER_SECOND,
    LEG_DOWN,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rlgym.lunar_lander.api.state import LunarLanderState


class DefaultObs(ObsBuilder[str, np.ndarray, LunarLanderState, tuple[str, int]]):
    """
    Builds the 8-dimensional observation vector from the current state.

    Corresponds to ObsBuilder in the RLGym API.
    """

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def build_obs(
        self, agents: List[str], state: LunarLanderState, shared_info: Dict[str, Any]
    ) -> Dict[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]]:
        obs = {}

        for agent in agents:
            pos = state.lander.position
            vel = state.lander.linearVelocity
            _agent_obs = np.array(
                [
                    (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                    (pos.y - (state.helipad_y + LEG_DOWN / SCALE))
                    / (VIEWPORT_H / SCALE / 2),
                    vel.x * (VIEWPORT_W / SCALE / 2) / TICKS_PER_SECOND,
                    vel.y * (VIEWPORT_H / SCALE / 2) / TICKS_PER_SECOND,
                    state.lander.angle,
                    20.0 * state.lander.angularVelocity / TICKS_PER_SECOND,
                    1.0 if state.legs[0].ground_contact else 0.0,
                    1.0 if state.legs[1].ground_contact else 0.0,
                ],
                dtype=np.float32,
            )
            obs[agent] = _agent_obs

        return obs

    def get_obs_space(self, agent: str) -> tuple[str, int]:
        return "real", 8
