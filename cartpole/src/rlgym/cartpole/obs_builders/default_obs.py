from typing import Any, Dict, List

from rlgym.api import ObsBuilder

from rlgym.cartpole.api import CartPoleState


class DefaultObs(ObsBuilder[str, list[float], CartPoleState, tuple[str, int]]):
    def get_obs_space(self, agent: str) -> tuple[str, int]:
        return "real", 4

    def build_obs(
        self, agents: List[str], state: CartPoleState, shared_info: Dict[str, Any]
    ) -> Dict[str, list[float]]:
        return {
            "cart": [
                state.cart.x,
                state.cart.x_dot,
                state.pole.theta,
                state.pole.theta_dot,
            ]
        }

    def reset(
        self,
        agents: List[str],
        initial_state: CartPoleState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass
