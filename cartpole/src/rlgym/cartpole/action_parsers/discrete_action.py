from typing import Any, Dict, List

from rlgym.api import ActionParser

from rlgym.cartpole.api import CartPoleState


class DiscreteAction(ActionParser[str, int, int, CartPoleState, tuple[str, int]]):
    """A basic action parser that takes a integer and gives it to the engine,
    it corresponds to the default action of the cartpole environment

    See https://gymnasium.farama.org/environments/classic_control/cart_pole/#action-space
    """

    def get_action_space(self, agent: str) -> tuple[str, int]:
        return "discrete", 2

    def parse_actions(
        self,
        actions: Dict[str, int],
        state: CartPoleState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, int]:
        return {"cart": actions["cart"]}

    def reset(
        self,
        agents: List[str],
        initial_state: CartPoleState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass
