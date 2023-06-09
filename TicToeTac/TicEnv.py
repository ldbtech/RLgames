import gymnasium
import numpy as np


class TicEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(9,))
        self.empty = 9
        self.max_steps = 20
        self.reset()

    def reset(self):
        self.action_space = gymnasium.spaces.Discrete(9)
        self.empty_cell = 9
        self.game_board = np.full(shape=(3, 3), fill_value=-1)
        self.done = False  # keep running
        self.steps = 0
        self.priority = -1

        return self.game_board

    """
        * Parameters: agents will have a form of [(col, row), (col, row)]
        * Priority can be Circle or X so to decide which agent can make the first move.
        * 0 is consider Circle, 1 is an X.
        * Agent Priority will be only called for first start of the game. 
        * Return Observation(game_board), reward, done, steps, info{agents}
    """

    # This one will need it during training to determine which one should go first.
    def agent_priority(self):
        priority = np.random.randint(0, 2)
        return [priority, 1 - priority]

    def step(self, agents, priority):
        print("Lovely Agents: ", agents)
        agent_a = agents[0]  # Fix: Access the first element of the agents list
        agent_b = agents[1]  # Fix: Access the second element of the agents list

        # Check if both agents have different coordinates
        if -1 in self.game_board:
            if (
                self.game_board[agent_a[0]][agent_a[1]] == -1
                and self.game_board[agent_b[0]][agent_b[1]] == -1
            ):
                if priority[0] == 0:  # Agent_a goes first.
                    self.game_board[agent_a[0]][agent_a[1]] = 0
                    self.game_board[agent_b[0]][agent_b[1]] = 1
                else:
                    self.game_board[agent_b[0]][agent_b[1]] = 1
                    self.game_board[agent_a[0]][agent_a[1]] = 0
                self.empty -= 2
                self.action_space.n = self.empty

        self.steps += 1
        # steps cannot exceed max_steps for each agent.
        # If a winner has been determined, no need to continue.
        winner = self.game_rule()
        if self.steps > self.max_steps or winner == 1 or winner == 0 or self.empty == 0:
            self.done = True
        (
            reward_a,
            reward_b,
        ) = self.calculate_rewards(
            winner=winner
        )  # Calculate rewards for each agent

        return self.game_board, self.done, [reward_a, reward_b], {}

    def game_rule(self):
        winning_coordinate = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]

        for comb in winning_coordinate:
            positions = [self.game_board[row][col] for row, col in comb]
            if all(position == 1 for position in positions):
                print("winning: X")
                return 1  # X
            elif all(position == 0 for position in positions):
                print("winning: 0")
                return 0  # Circle
        return None

    def calculate_rewards(self, winner):
        if winner == 0:
            print("Winner is 0")
            reward = [1, -1]
        elif winner == 1:
            print("Winner X")
            reward = [-1, 1]
        elif self.empty == 0:
            print("Draw")
            reward = [0, 0]
        else:
            print("Weirdo")
            reward = [-0.1, -0.1]

        return reward[0], reward[1]  # No winner, return equal rewards
