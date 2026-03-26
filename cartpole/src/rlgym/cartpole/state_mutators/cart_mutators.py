import random
from typing import Any, Dict

from rlgym.api import StateMutator

from rlgym.cartpole.api import CartPoleState


class CartMutator(StateMutator[CartPoleState]):
    def __init__(self, deviation_from_center: float = 0.05) -> None:
        self.deviation_from_center = deviation_from_center

    def apply(self, state: CartPoleState, shared_info: Dict[str, Any]) -> None:
        state.cart.x = random.uniform(
            -self.deviation_from_center, self.deviation_from_center
        )


class CartImpulseMutator(StateMutator[CartPoleState]):
    def __init__(self, max_impulse: float = 0.05) -> None:
        self.max_impulse = max_impulse

    def apply(self, state: CartPoleState, shared_info: Dict[str, Any]) -> None:
        state.cart.x_dot = random.uniform(-self.max_impulse, self.max_impulse)
