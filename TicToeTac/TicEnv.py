import gymnasium
import numpy as np


class TicEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(9,))
        self.empty_cell = 9
        self.max_steps = 5
        self.reset()

    def reset(self):
        self.action_space = gymnasium.spaces.Discrete(9)
        self.empty_cell = 9
        self.game_board = np.full(shape=(3, 3), fill_value=-1)
        self.done = False  # keep running
        self.steps = 0
        self.priority = -1
        self.reward = []

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
        agent_a = agents[0]
        agent_b = agents[1]
        # WE HAVE TO CHECK BOTH AGENTS DONT HAVE THE SAME COORDINATES
        if -1 in self.game_board:
            if (
                self.game_board[agent_a[0]][agent_a[1]] == -1
                and self.game_board[agent_b[0]][agent_b[1]] == -1
            ):
                if priority[0] == 0:  # Agent_a go first.
                    self.game_board[agent_a[0]][agent_a[1]] = 0
                    self.game_board[agent_b[0]][agent_b[1]] = 1
                else:
                    self.game_board[agent_b[0]][agent_b[1]] = 1
                    self.game_board[agent_a[0]][agent_a[1]] = 0
        """
        if self.priority == -1:
            self.priority = self.agent_priority()
        if self.priority[0] == 0:  # Agent A make first move.
            self.
        elif self.priority[1] == 0:  # Agent B make first move.
            pass
        """

        self.steps += 1
        # steps cannot exceeds max_steps each agents have to take.
        # Also, if winner have been determined then no need to continue.
        if self.steps > 20 or self.game_rule() == 1 or self.game_rule() == 0:
            self.done = True

        return self.game_board, self.done, self.reward, {}

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
                self.reward.append([1, 0])
                return 1  # X
            elif all(position == 0 for position in positions):
                print("winning: 0")
                self.reward.append([0, 1])
                return 0  # Circle

        return None

    def render(self):
        pass
