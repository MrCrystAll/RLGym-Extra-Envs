import random
from typing import Any, Dict

from rlgym.api import StateMutator

from rlgym.cartpole.api import CartPoleState


class PoleMutator(StateMutator[CartPoleState]):
    def __init__(self, default_max_angle: float = 0.05) -> None:
        self.default_max_angle = default_max_angle

    def apply(self, state: CartPoleState, shared_info: Dict[str, Any]) -> None:
        state.pole.theta = random.uniform(
            -self.default_max_angle, self.default_max_angle
        )
