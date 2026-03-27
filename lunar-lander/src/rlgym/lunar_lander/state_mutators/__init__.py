from typing import Any, Dict

import numpy as np
from rlgym.api import StateMutator

from rlgym.lunar_lander.api import LunarLanderState

from .state_mutator import (
    WindMutator,
    InitialBumpMutator,
    TurbulenceMutator,
    GravityMutator,
)


class DefaultMutator(StateMutator[LunarLanderState]):
    def __init__(
        self,
        seed: int = 123,
        gravity: float = -10.0,
        wind_power: float = 15,
        turbulence_power: float = 1.5,
    ) -> None:
        self.wind_mut = WindMutator(seed=seed, wind_max_speed=wind_power)
        self.turbulence_mut = TurbulenceMutator(
            seed=seed, max_perturbation=turbulence_power
        )
        self.gravity_mut = GravityMutator(gravity=gravity)
        self.bump_mut = InitialBumpMutator()

    def apply(self, state: LunarLanderState, shared_info: Dict[str, Any]) -> None:
        self.wind_mut.apply(state, shared_info)
        self.turbulence_mut.apply(state, shared_info)
        self.gravity_mut.apply(state, shared_info)
        self.bump_mut.apply(state, shared_info)
