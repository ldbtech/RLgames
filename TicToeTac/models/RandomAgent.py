import random
import numpy as np
import sys
import os

"""
Tomorrow goal: to see how i can make the agent understand that some values are occupied. 
also decrease avialable cells in tic tac toe. The Agent cannot pick occupied ones ..etc. 
Once done with random Agent, then we can develop Deep Q-learning to solve the environment. 

--------------------------------[Algorithms to implement]--------------------------------
* Deep Q with Replay functionality. 
* Deep Double Q Learning with Replay Functionality. 
* Actor Critic and Advantage Actor Critic A2C.

Once training done, deploy and try it playing against it. Make a complete game can be played on the web.
Use JS to deploy after trying Pygame.
"""
# Get the parent directory of the Models folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from TicEnv import TicEnv


class RandomAgent:
    def __init__(self, env):
        self.done = False
        self.env = env

    def generate_random_coords(self):
        coords = []
        for _ in range(2):
            col = np.random.randint(0, 3)
            row = np.random.randint(0, 3)
            coords.append((col, row))

        while coords[0] == coords[1]:
            coords[1] = (np.random.randint(0, 3), np.random.randint(0, 3))

        return coords

    def run(self):
        obs = self.env.reset()
        priority = self.env.agent_priority()
        reward = []

        while not self.done:
            agents = self.generate_random_coords()

            obs, self.done, step_reward, info = self.env.step(
                agents=agents, priority=priority
            )
            reward.append(step_reward)

        print("Last Act:\n", obs)
        print("Final Rewards:", reward)


env = TicEnv()
agent = RandomAgent(env=env)

agent.run()
