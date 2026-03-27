from typing import Any, Dict, List

from rlgym.api import RewardFunction

from rlgym.lunar_lander.api.state import LunarLanderState

from .angle_punishment import AnglePunishment
from .distance_to_landing_pad import DistanceToLandingPadPunishment
from .speed_to_landing_pad import SpeedToLandingPadPunishment
from .leg_hit_reward import LegHitReward
from .main_engine_punishment import MainEnginePunishment
from .lateral_engine_punishment import LateralEnginePunishment
from .landing_reward import LandingReward


class DefaultReward(RewardFunction[str, LunarLanderState, float]):
    def __init__(self) -> None:
        self._prev_shaping: dict[str, float] = {}

        self.dist_to_pad = DistanceToLandingPadPunishment()
        self.speed_to_pad = SpeedToLandingPadPunishment()
        self.angle = AnglePunishment()
        self.landing_leg = LegHitReward()
        self.main_engine = MainEnginePunishment()
        self.lateral_engine = LateralEnginePunishment()
        self.landing = LandingReward()

    def reset(
        self,
        agents: List[str],
        initial_state: LunarLanderState,
        shared_info: Dict[str, Any],
    ) -> None:
        self._prev_shaping = {agent: 0.0 for agent in agents}

        self.dist_to_pad.reset(agents, initial_state, shared_info)
        self.speed_to_pad.reset(agents, initial_state, shared_info)
        self.angle.reset(agents, initial_state, shared_info)
        self.landing_leg.reset(agents, initial_state, shared_info)
        self.main_engine.reset(agents, initial_state, shared_info)
        self.lateral_engine.reset(agents, initial_state, shared_info)
        self.landing.reset(agents, initial_state, shared_info)

    def get_rewards(
        self,
        agents: List[str],
        state: LunarLanderState,
        is_terminated: Dict[str, bool],
        is_truncated: Dict[str, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[str, float]:
        _dist_to_pad = self.dist_to_pad.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        _speed_to_pad = self.speed_to_pad.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        _angle = self.angle.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        _landing_leg = self.landing_leg.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        _main_engine = self.main_engine.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        _lateral_engine = self.lateral_engine.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )

        _rewards = {}

        for agent in agents:
            _shaping = (
                100 * _dist_to_pad[agent]  # Dist to pad is negative
                + 100 * _speed_to_pad[agent]  # Speed to pad is negative
                + 100 * _angle[agent]  # Angle is negative
                + _landing_leg[
                    agent
                ]  # Landing leg can be both, but is positive by default
            )

            reward = _shaping - self._prev_shaping[agent]

            self._prev_shaping[agent] = _shaping

            reward += _main_engine[agent]  # Main engine is negative
            reward += _lateral_engine[agent]  # Lateral engine is negative

            _rewards[agent] = reward

        return _rewards
