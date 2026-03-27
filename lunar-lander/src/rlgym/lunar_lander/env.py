import numpy as np
from rlgym.api import RLGym

from rlgym.lunar_lander.action_parsers.continuous_action import ContinuousAction
from rlgym.lunar_lander.action_parsers.discrete_action import DiscreteAction
from rlgym.lunar_lander.api.state import LunarLanderState
from rlgym.lunar_lander.done_conditions.crash_landing import CrashLandingCondition
from rlgym.lunar_lander.done_conditions.timeout_condition import TimeoutCondition
from rlgym.lunar_lander.obs_builders.default_obs import DefaultObs
from rlgym.lunar_lander.renderers.pygame_renderer import PygameRenderer
from rlgym.lunar_lander.reward_functions import DefaultReward
from rlgym.lunar_lander.state_mutators import DefaultMutator
from rlgym.lunar_lander.transition_engines.default_engine import DefaultEngine


class LunarLander(
    RLGym[
        str,
        np.ndarray,
        np.ndarray | int,
        np.ndarray,
        float,
        LunarLanderState,
        tuple[str, int],
        tuple[str, int],
    ],
):
    """A reproduction of the gymnasium environment"""

    def __init__(
        self,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15,
        turbulence_power: float = 1.5,
        seed: int = 123,
    ):
        if continuous:
            raise NotImplementedError("Continuous is not implemented yet. Sorry :/")

        action_parser = ContinuousAction() if continuous else DiscreteAction()

        super().__init__(
            DefaultMutator(
                seed, gravity, wind_power if enable_wind else 0, turbulence_power
            ),
            DefaultObs(),
            action_parser,
            DefaultReward(),
            DefaultEngine(),
            CrashLandingCondition(),
            TimeoutCondition(),
            None,
            PygameRenderer(),
        )
