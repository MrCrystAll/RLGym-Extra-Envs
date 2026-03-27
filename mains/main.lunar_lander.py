import random

from rlgym.lunar_lander.api.common_values import AGENT_NAME
from rlgym.lunar_lander.env import LunarLander

if __name__ == "__main__":
    env = LunarLander()

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
