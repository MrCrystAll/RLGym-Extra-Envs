from typing import Any, Dict

from rlgym.api import StateMutator

from rlgym.cartpole.api import CartPoleState

from .cart_mutators import CartMutator, CartImpulseMutator
from .pole_mutators import PoleMutator


class DefaultMutator(StateMutator[CartPoleState]):
    """The default state for the CartPole environment"""

    def __init__(self) -> None:
        self._cart_mut = CartMutator()
        self._cart_imp_mut = CartImpulseMutator()
        self._pole_mut = PoleMutator()

    def apply(self, state: CartPoleState, shared_info: Dict[str, Any]) -> None:
        self._cart_mut.apply(state, shared_info)
        self._cart_imp_mut.apply(state, shared_info)
        self._pole_mut.apply(state, shared_info)
