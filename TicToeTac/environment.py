import pygame
import numpy as np
import gymnasium
import pickle as pk


class TicTacToe(gymnasium.Env):
    """
    ------------------------[Deployment]--------------------------
    * This environment will be used after we trained the model.
    """

    def __init__(self):
        # Initialize the game
        pygame.init()
        width, height = 700, 700
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Tic Tac Toe")
        self.clock = pygame.time.Clock()

        # Game variables
        self.running = True
        self.draw_circle = False
        self.line_color = (0, 0, 0)
        self.game_board = np.full((3, 3), -1)
        self.clicked = False
        self.reset()
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(9,)
        )  # states of the game.
        self.action_space = gymnasium.spaces.Discrete(
            9
        )  # action of the game at the initial.
        self.empty = (
            9  # empty cell equal to the number of the action when game just started.
        )

    def reset(self):
        self.action_space = gymnasium.spaces.Discrete(9)
        self.empty = 9
        self.draw_circle = False
        self.game_board = np.full((3, 3), -1)
        self.running = True
        self.clicked = False
        self.game_ended = False
        self.done = False

        return self.game_board

    # this is used for reinforcement learning agents
    def steps(self, actions):
        agent_1 = (actions[0] % 3, actions[0] // 3)
        agent_2 = (actions[1] % 3, actions[1] // 3)
        reward = []

        print("agent_1: ", agent_1, " Agent_2: ", agent_2)

        if (
            self.game_board[agent_1[0]][agent_1[1]] == -1
            and self.game_board[agent_2[0]][agent_2[1]] == -1
        ):
            self.game_board[agent_1[0]][agent_1[1]] = 1
            self.game_board[agent_2[0]][agent_2[1]] = 0

            self.empty -= 2
            self.action_space.n = self.empty
            print("Board: ", self.game_board)
            winner = self.winner_check()

            if winner == 1:
                print("Winner is 1")
                reward = [1, -1]  # Positive reward for agent 1 and negative for agent 2
                self.done = True
            elif winner == 0:
                print("Winner is 0")
                reward = [-1, 1]
                self.done = True
            elif self.empty == 0:
                print("Draw")
                reward = [0, 0]
                self.done = True
            else:
                reward = [-0.1, -0.1]
                self.done = False
        print("Hello: ")
        obs = self.game_board.flatten()
        print("end")

        return obs, reward, self.done, {}

    def step(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        winner = self.handle_mouse_click(mouse_pos)
                        if winner == 0 or winner == 1 or -1:
                            print("Game ends, start new game")
                            self.game_ended = True
                        self.clicked = True
                        if self.draw_circle:
                            self.draw_circle = False
                        else:
                            self.draw_circle = True

            self.screen.fill("purple")
            self.draw_board(line_color=self.line_color)
            if self.clicked:
                self.draw_shape(mouse_pos, draw_circle=self.draw_circle)

            pygame.display.flip()
            self.clock.tick(60)  # 60 fps
            keys = pygame.key.get_pressed()
            if self.game_ended and keys[pygame.K_r]:  # if game ends or user clicked R
                print("Starting a new game")
                self.reset()
                self.game_ended = False

        pygame.quit()
        # this is where the game is being played.

    def draw_board(self, line_color):
        width, height = self.screen.get_size()
        cell_size = width // 3

        # Drawing vertical lines
        for i in range(1, 3):
            pygame.draw.line(
                self.screen, line_color, (i * cell_size, 0), (i * cell_size, height), 7
            )

        # Drawing horizontal lines
        for i in range(1, 3):
            pygame.draw.line(
                self.screen, line_color, (0, i * cell_size), (width, i * cell_size), 7
            )

    def draw_previous_shapes(self):
        cell_size = self.screen.get_width() // 3
        if 1 in self.game_board or 0 in self.game_board:
            for row in range(3):
                for col in range(3):
                    if self.game_board[row][col] == 1:  # circle shape
                        center_x = col * cell_size + cell_size // 2
                        center_y = row * cell_size + cell_size // 2
                        radius = cell_size // 2 - 10
                        pygame.draw.circle(
                            self.screen,
                            (152, 251, 152),
                            (center_x, center_y),
                            radius,
                            5,
                        )
                        pygame.draw.circle(
                            self.screen, "purple", (center_x, center_y), radius - 5, 10
                        )
                    if self.game_board[row][col] == 0:  # circle X
                        top_left = (col * cell_size + 10, row * cell_size + 10)
                        bottom_right = (
                            col * cell_size + cell_size - 10,
                            row * cell_size + cell_size - 10,
                        )
                        pygame.draw.line(
                            self.screen, (255, 0, 0), top_left, bottom_right, 5
                        )
                        pygame.draw.line(
                            self.screen,
                            (255, 0, 0),
                            (bottom_right[0], top_left[1]),
                            (top_left[0], bottom_right[1]),
                            5,
                        )

    def draw_shape(self, pos, draw_circle):
        cell_size = self.screen.get_width() // 3
        cell_col = pos[0] // cell_size
        cell_row = pos[1] // cell_size
        self.draw_previous_shapes()
        if self.game_board[cell_row][cell_col] == -1:
            if draw_circle:  # draw new shape
                cell_size = self.screen.get_width() // 3
                radius = cell_size // 2 - 10
                center_x = (pos[0] // cell_size) * cell_size + cell_size // 2
                center_y = (pos[1] // cell_size) * cell_size + cell_size // 2

                pygame.draw.circle(
                    self.screen, (152, 251, 152), (center_x, center_y), radius, 5
                )
                pygame.draw.circle(
                    self.screen, "purple", (center_x, center_y), radius - 5, 10
                )
            else:
                top_left = (cell_col * cell_size + 10, cell_row * cell_size + 10)
                bottom_right = (
                    cell_col * cell_size + cell_size - 10,
                    cell_row * cell_size + cell_size - 10,
                )

                pygame.draw.line(self.screen, (255, 0, 0), top_left, bottom_right, 5)
                pygame.draw.line(
                    self.screen,
                    (255, 0, 0),
                    (bottom_right[0], top_left[1]),
                    (top_left[0], bottom_right[1]),
                    5,
                )

    def handle_mouse_click(self, pos):
        # Handle the logic for updating the game state based on the mouse click position
        # For example, you can determine which cell of the board was clicked and update the board array accordingly
        cell_size = self.screen.get_width() // 3
        cell_col = pos[0] // cell_size
        cell_row = pos[1] // cell_size
        if self.game_board[cell_row][cell_col] == -1:
            self.game_board[cell_row][cell_col] = 1 if self.draw_circle else 0
            self.empty -= 1
            self.action_space.n = self.empty
            action_index = cell_row * 3 + cell_col
            # self.action_space.nvec[action_index] = -1
            print("Board: \n", self.game_board)
            winner = self.winner_check()
            if winner == 1:
                print("Winner: Circle")
                return 1
            if winner == 0:
                print("Winner X")
                return 0

    def winner_check(self):
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
            print("Position: ", positions)
            if all(position == 1 for position in positions):
                return 1  # Circle
            elif all(position == 0 for position in positions):
                return 0  # X

        return None


c = TicTacToe()
done = False
observation = c.reset()
import time

while not done:
    action_smaple = np.random.randint(0, 8, size=2)
    print(action_smaple)
    obs, reward, done, info = c.steps(action_smaple)
    print("Done: ", done)
    # time.sleep(3)

    print("Reward: ", reward)
