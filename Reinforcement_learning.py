import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        return self.env.action_space.sample()
    
env = gym.make("Taxi-v3", render_mode="rgb_array")
agent = RandomAgent(env)
env.reset()
env.s = 123
reward = 0
penality = 0
frames = []
done = False
epoches = 0

while not done:
    state = env.s
    action = agent.get_action(state)
    state_resu = env.step(action)
    state =state_resu[0]
    reward =state_resu[1]
    done =state_resu[2]

    if(reward == -10):
        penality += 1

    frames.append({
        'state': state,
        'action': action,
        'reward': reward 
    })
    epoches += 1
print("the epoches is: ", epoches)
print("the penality is: ", penality)
print("the reward is: ", reward)

renderIm = env.render()

plt.imshow(renderIm)
plt.show()

display.display(plt.gcf())
display.clear_output(True)
