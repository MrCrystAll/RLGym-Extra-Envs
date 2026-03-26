from typing import Any, Dict, List

from rlgym.api import DoneCondition

from rlgym.cartpole.api.state import CartPoleState


class DefaultTruncation(DoneCondition[str, CartPoleState]):
    """The truncation condition used by the default CartPole environment

    See https://gymnasium.farama.org/environments/classic_control/cart_pole/#episode-end (3)
    """

    def __init__(self) -> None:
        self.max_time = 500
        self.ticks_until_truncation = 0

    def is_done(
        self, agents: List[str], state: CartPoleState, shared_info: Dict[str, Any]
    ) -> Dict[str, bool]:
        print(self.ticks_until_truncation)
        self.ticks_until_truncation += 1
        return {"cart": self.ticks_until_truncation > self.max_time}

    def reset(
        self,
        agents: List[str],
        initial_state: CartPoleState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.ticks_until_truncation = 0
