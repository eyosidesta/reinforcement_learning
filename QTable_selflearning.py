import gymnasium as gym

def main():
    env = gym.make('Taxi-V3', render_mode='rgb_array')

    training_epoch = 1000
    max_each_epoch_step = 100

    action = env.action_space.n

    explore = 1.0

    for _ in training_epoch():
        state = env.reset()
        for _ in max_each_epoch_step():
            if explore > 0.5:
                action = env.action_space.sample()
            else:
                action = env.action_space(qtable[state: ,])

            take_action = env.step(action)
            state = take_action[0]
            

    
