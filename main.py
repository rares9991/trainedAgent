import numpy as np
import gym
import random
import pygame


def main():
    # create Taxi environment
    env = gym.make('Taxi-v3', render_mode="ansi")

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99  # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state[0], :])

            # take action and observe reward
            new_state = env.step(action)

            # Q-learning algorithm
            qtable[state[0], action] = qtable[state[0], action] + learning_rate * (
                        new_state[1] + discount_rate * np.max(qtable[new_state[0], :]) - qtable[state[0], action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state[0], :])
        new_state = env.step(action)
        rewards += new_state[1]
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if new_state[2] == True:
            break

    env.close()


if __name__ == "__main__":
    main()