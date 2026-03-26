from typing import Any, Dict
import numpy as np

from rlgym.api import StateMutator

from rlgym.lunar_lander.state import LunarLanderState


class InitialBumpMutator(StateMutator[LunarLanderState]):
    def __init__(self, seed: int = 123, bump_force: float = 1000) -> None:
        self.rng = np.random.RandomState(seed)
        self.bump_force = bump_force

    def apply(self, state: LunarLanderState, shared_info: Dict[str, Any]) -> None:
        state.lander.ApplyForceToCenter(
            (
                self.rng.uniform(-self.bump_force, self.bump_force),
                self.rng.uniform(-self.bump_force, self.bump_force),
            ),
            True,
        )


class WindMutator(StateMutator[LunarLanderState]):
    def __init__(self, seed: int = 123, wind_max_speed: float = 15) -> None:
        self.rng = np.random.RandomState(seed)
        self.wind_max_speed = wind_max_speed

    def apply(self, state: LunarLanderState, shared_info: Dict[str, Any]) -> None:
        state.config.wind_power = self.rng.uniform(-1, 1) * self.wind_max_speed
        print(f"Wind power: {state.config.wind_power}")
