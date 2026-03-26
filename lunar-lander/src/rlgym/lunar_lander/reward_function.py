from typing import Any, Dict, List

import numpy as np

from rlgym.api import RewardFunction

from rlgym.lunar_lander.common_values import (
    AGENT_NAME,
    TICKS_PER_SECOND,
    LEG_DOWN,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rlgym.lunar_lander.state import LunarLanderState


class LunarLanderRewardFunction(RewardFunction[str, LunarLanderState, float]):
    """
    Computes the per-step reward using potential-based shaping.

    Corresponds to RewardFunction in the RLGym API.
    """

    def __init__(self) -> None:
        self._prev_shaping = None

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        pos = initial_state.lander.position
        vel = initial_state.lander.linearVelocity

        pos_x = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        pos_y = (pos.y - (initial_state.helipad_y + LEG_DOWN / SCALE)) / (
            VIEWPORT_H / SCALE / 2
        )

        vel_x = vel.x * (VIEWPORT_W / SCALE / 2) / TICKS_PER_SECOND
        vel_y = vel.y * (VIEWPORT_H / SCALE / 2) / TICKS_PER_SECOND
        self._prev_shaping = (
            -100 * np.sqrt(pos_x**2 + pos_y**2)
            - 100 * np.sqrt(vel_x**2 + vel_y**2)
            - 100 * abs(initial_state.lander.angle)
            + 10 * float(initial_state.legs[0].ground_contact)
            + 10 * float(initial_state.legs[1].ground_contact)
        )

    def get_rewards(
        self,
        agents: List[str],
        state: LunarLanderState,
        is_terminated: Dict[str, bool],
        is_truncated: Dict[str, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[str, float]:
        m_power = shared_info.get("m_power", 0.0)
        s_power = shared_info.get("s_power", 0.0)
        pos = state.lander.position
        vel = state.lander.linearVelocity

        pos_x = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        pos_y = (pos.y - (state.helipad_y + LEG_DOWN / SCALE)) / (
            VIEWPORT_H / SCALE / 2
        )

        vel_x = vel.x * (VIEWPORT_W / SCALE / 2) / TICKS_PER_SECOND
        vel_y = vel.y * (VIEWPORT_H / SCALE / 2) / TICKS_PER_SECOND

        shaping = (
            -100 * np.sqrt(pos_x**2 + pos_y**2)
            - 100 * np.sqrt(vel_x**2 + vel_y**2)
            - 100 * abs(state.lander.angle)
            + 10 * float(state.legs[0].ground_contact)
            + 10 * float(state.legs[1].ground_contact)
        )

        reward = 0.0
        prev = self._prev_shaping
        reward = shaping - prev
        self._prev_shaping = shaping

        reward -= m_power * 0.30
        reward -= s_power * 0.03

        # Terminal bonuses (applied externally via DoneCondition, but we
        # handle the scalar bonus here so the reward component is self-contained)
        if is_terminated[AGENT_NAME]:
            if state.game_over or abs(pos_x) >= 1.0:
                reward = -100.0
            elif not state.lander.awake:
                reward = +100.0

        return {AGENT_NAME: reward}
