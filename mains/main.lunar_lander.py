import random

from rlgym.lunar_lander.action_parser import LunarLanderDiscreteActionParser
from rlgym.lunar_lander.common_values import AGENT_NAME
from rlgym.lunar_lander.obs_builder import LunarLanderObsBuilder
from rlgym.lunar_lander.state_mutator import WindMutator
from rlgym.lunar_lander.reward_function import LunarLanderRewardFunction
from rlgym.lunar_lander.renderer import LunarLanderRenderer
from rlgym.lunar_lander.terminal_condition import (
    LunarLanderTermination,
    LunarLanderTruncation,
)
from rlgym.lunar_lander.engine import LunarLanderTransitionEngine

from rlgym.api import RLGym

if __name__ == "__main__":
    env = RLGym(
        obs_builder=LunarLanderObsBuilder(),
        action_parser=LunarLanderDiscreteActionParser(),
        state_mutator=WindMutator(),
        reward_fn=LunarLanderRewardFunction(),
        renderer=LunarLanderRenderer(),
        termination_cond=LunarLanderTermination(),
        truncation_cond=LunarLanderTruncation(),
        transition_engine=LunarLanderTransitionEngine(),
    )

    running = True

    print("Running env")
    while running:
        try:
            obs = env.reset()

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not truncated[AGENT_NAME] and not terminated[AGENT_NAME]:
                env.render()

                # agent_actions = act_parser.sample(env.agents)
                agent_actions = {AGENT_NAME: random.randint(0, 3)}

                obs, reward, terminated, truncated = env.step(agent_actions)

        except KeyboardInterrupt:
            print("Stopping")
            running = False
            break

    env.close()
