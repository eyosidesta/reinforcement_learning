
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

class TrainedAgent:
    def __init__(self, env):
        self.env = env

    def train_agent(self):
        training_epoch = 1000

        # The agent tries trial and error until explore becomes less than random.uniform(0,1) in order to take the action or to determine what to do.
        explore = 1.0
        # The agent will begin to exploit or move based on the data from the qtable rather than moving by guessing.
        # The explore value is minimized based on the explore_changer. This will increase performance and minimize or decrease the penality.
        minimize_explorer= 0.005

        
        state_number = self.env.observation_space.n
        action_number = self.env.action_space.n
        # assign 0 to each value in the q-table at first. Over time, the agent will learn, and each qtable value will be changed by qvalue, 
        # enabling the agent to take the necessary action. 
        qtable = np.zeros((state_number, action_number))

        qlearning_rate = 0.9
        discount_rate = 0.8

        # train the agent with reinforcement approach, it will learn by try and error, trhough time the agent will adapt
        # what kind of action needs to take in every step instead of doing it randomly, this will increase the performance.
        for episode in range(training_epoch):
            # reset the environment
            state = self.env.reset()
            done = False

            while not done:
                # explore untile explore value is less than random.uniform(0, 1), after this point the agent will start to move intelligently
                if random.uniform(0,1) < explore:
                    action = self.env.action_space.sample()
                else:
                    # the agent will start to act according to the qtable values (state)
                    actual_state = state[0] if isinstance(state, tuple) else state
                    action = np.argmax(qtable[actual_state,:])

                # take action and observe reward
                take_action = self.env.step(action)
                update_state = take_action[0]
                reward = take_action[1]
                done = take_action[2]
                print(f"state: {state}, action: {action}, update_state: {update_state}")
                actual_state = state[0] if isinstance(state, tuple) else state
                qtable[actual_state, action] = qtable[actual_state, action] + qlearning_rate * (reward + discount_rate * np.max(qtable[update_state]) - qtable[actual_state, action])
                
                # update the state
                state = update_state

            # Decrease explore by minimize_explorer value for each episode
            explore = np.exp(-minimize_explorer*episode)

        print(f"Training completed over {training_epoch} episodes")
        return qtable

    def main(self):

        qtable = self.train_agent()

        # Test the trained agent
        state = self.env.reset()
        done = False
        rewards = 0
        penality = 0
        count_steps = 0

        while not done:
            actual_state = state[0] if isinstance(state, tuple) else state
            action = np.argmax(qtable[actual_state,:])
            take_action = self.env.step(action)
            update_state = take_action[0]
            reward = take_action[1]
            done = take_action[2]
            
            if reward == -10:
                penality += 1
            rewards = rewards + reward
            # update state
            state = update_state

            count_steps += 1
            
        renderVal = self.env.render()
        print(f"Trained with Reinforcement Learning")
        print("Steps {}".format(count_steps))
        print(f"reward: {rewards}")
        print(f"penalty: {penality}")
        plt.figure("Trained Agent")
        plt.title(label="Trained Agent")
        plt.imshow(renderVal)
        plt.show()
        self.env.close()

# create Taxi environment
env = gym.make('Taxi-v3', render_mode="rgb_array")
trainedAgent = TrainedAgent(env)
trainedAgent.main()

