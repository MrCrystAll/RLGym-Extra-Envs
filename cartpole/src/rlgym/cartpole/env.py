from rlgym.api import RLGym

from rlgym.cartpole.action_parsers import DiscreteAction
from rlgym.cartpole.api import CartPoleState
from rlgym.cartpole.done_conditions import DefaultTruncation, DefaultTermination
from rlgym.cartpole.obs_builders import DefaultObs
from rlgym.cartpole.renderers import PygameRenderer
from rlgym.cartpole.reward_functions import EpisodeLengthReward
from rlgym.cartpole.state_mutators import DefaultMutator
from rlgym.cartpole.transition_engines import DefaultEngine


class CartPole(
    RLGym[
        str,
        list[float],
        int,
        int,
        float,
        CartPoleState,
        tuple[str, int],
        tuple[str, int],
    ]
):
    """Just a shortcut for a reproduction of the Gymnasium environment,
    use the RLGym class if you want to give custom components
    """

    def __init__(self, sutton_barto_reward: bool = False):
        super().__init__(
            DefaultMutator(),
            DefaultObs(),
            DiscreteAction(),
            EpisodeLengthReward(sutton_barto_reward=sutton_barto_reward),
            DefaultEngine(),
            DefaultTermination(),
            DefaultTruncation(),
            None,
            PygameRenderer(),
        )
