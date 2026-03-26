from typing import Any, Dict, List

from rlgym.api import DoneCondition

from rlgym.cartpole.api.state import CartPoleState


class DefaultTermination(DoneCondition[str, CartPoleState]):
    """The termination condition used by the default CartPole environment

    See https://gymnasium.farama.org/environments/classic_control/cart_pole/#episode-end (1 and 2)
    """

    def is_done(
        self, agents: List[str], state: CartPoleState, shared_info: Dict[str, Any]
    ) -> Dict[str, bool]:
        return {
            "cart": state.cart.x < -state.config.x_threshold
            or state.cart.x > state.config.x_threshold
            or state.pole.theta < -state.config.theta_threshold
            or state.pole.theta > state.config.theta_threshold
        }

    def reset(
        self,
        agents: List[str],
        initial_state: CartPoleState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass
