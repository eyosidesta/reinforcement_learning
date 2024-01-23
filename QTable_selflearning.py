import numpy as np
import gymnasium as gym

def main():
    env = gym.make('Taxi-V3', render_mode='rgb_array')

    training_epoch = 1000
    max_each_epoch_step = 100

    space = env.observation_space.n
    action = env.action_space.n

    q_learning_rate = 0.9
    # The agent tries trial and error until explore becomes less than random.uniform(0,1) in order to take the action or to determine what to do.
    explore = 1.0

    # The agent will begin to exploit or move based on the data from the qtable rather than moving by guessing.
    # The explore value is minimized based on the explore_changer. This will increase performance and minimize or decrease the penality.

    explore_changer = 0.005

    for training in training_epoch():
        state = env.reset()
        for steps in max_each_epoch_step():
            if explore > 0.5:
                action = env.action_space.sample()
            else:
                action = env.action_space(qtable[state: ,])

            take_action = env.step(action)
            state = take_action[0]
            reward = take_action[1]
            done = take_action[1]
            

    
    done = False
    reward = 0
    penality = 0
    count_steps = 0
    rewards = 0

    while not done:
        take_action = env.step(action)
        state = take_action[0]
        reward = take_action[1]
        done = take_action[2]
        if (reward == -10):
            penality += 1
        rewards = rewards + reward
        count_steps += 1

    env.render()
    print("Steps: ", steps)
    print("Penality: ", penality)
    print("reward: ", reward)
