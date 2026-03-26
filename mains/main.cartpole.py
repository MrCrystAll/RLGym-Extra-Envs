import random
import time

from rlgym.cartpole import CartPole

if __name__ == "__main__":
    env = CartPole(sutton_barto_reward=False)

    running = True

    print("Running env")
    while running:
        try:
            obs = env.reset()

            action_space = env.action_space("cart")

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not truncated["cart"] and not terminated["cart"]:
                env.render()

                agent_actions = {"cart": 1}

                obs, reward, terminated, truncated = env.step(agent_actions)

        except KeyboardInterrupt:
            print("Stopping")
            running = False
            break

    env.close()
