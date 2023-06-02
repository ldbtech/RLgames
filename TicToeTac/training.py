from TicEnv import TicEnv
from models.DeepQ import DeepQ


def training():
    epsiodes = 1000
    env = TicEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent_a = DeepQ(
        state_size=state_size,
        action_size=action_size,
        epsilon_decay=0.999,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_min=0.01,
        memory_size=500,
    )
    agent_a.model.compile(
        optimizer="adam", loss="mse"
    )  # Add compile method for agent_a

    agent_b = DeepQ(
        state_size=state_size,
        action_size=action_size,
        epsilon_decay=0.999,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_min=0.01,
        memory_size=500,
    )
    agent_b.model.compile(
        optimizer="adam", loss="mse"
    )  # Add compile method for agent_b

    batch_size = 32
    priority = env.agent_priority()
    all_rewards_a = []
    all_rewards_b = []

    for ep in range(epsiodes):
        state = env.reset()
        state = state.reshape(1, state_size)
        done = False
        total_reward_a = 0
        total_reward_b = 0
        while not done:
            action_a = agent_a.takeAction(state=state)
            action_b = agent_b.takeAction(state=state)

            while action_a == action_b:
                action_a = agent_a.takeAction(state=state)
                action_b = agent_b.takeAction(state=state)
            print("Action_a: ", action_a, " Action_b: ", action_b)
            agent_a_ = (action_a[0][0], action_a[0][1])
            agent_b_ = (action_b[0][0], action_b[0][1])

            print("Agent A : {}, agent B: {}".format(action_a, action_b))
            next_state, done, reward, info = env.step(
                [agent_a_, agent_b_], priority=priority
            )
            next_state = next_state.reshape(1, state_size)
            transition_a = (state, action_a, reward[0], next_state, done)
            agent_a.remember(transition_a)
            transition_b = (state, action_b, reward[1], next_state, done)
            agent_b.remember(transition_b)

            state = next_state
            print("reward: ", reward)
            total_reward_a += reward[0]
            total_reward_b += reward[1]

            if len(agent_a.memory) > batch_size:
                agent_a.replay(batch_size=batch_size)
            if len(agent_b.memory) > batch_size:
                agent_b.replay(batch_size=batch_size)

        if ep % 100 == 0:
            print(
                "Episodes: {}, total_reward A: {}, total_reward B: {}".format(
                    ep, total_reward_a, total_reward_b
                )
            )
        all_rewards_a.append(total_reward_a)
        all_rewards_b.append(total_reward_b)

    agent_a.save_model("SavedModel_agentA")
    agent_b.save_model("SavedModel_agentB")
    return all_rewards_a, all_rewards_b


def main():
    training()


if __name__ == "__main__":
    main()
