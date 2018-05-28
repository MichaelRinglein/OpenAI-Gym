import gym

env = gym.make('CartPole-v0') # determine enviroment

print('Initial Observation')
observation = env.reset() # Reset enviroment to default starting
print(observation) # will show something like [-0.02501606  0.02719068  0.0428136   0.04119534], the pole standing up


for t in range(20): # Render enviroment over 1000 timesteps

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action) # Take some sort of random action, 

    print('At step ', t)
    print('\n')
    print('observation')
    print(observation)
    print('\n')

    print('reward')
    print(reward)
    print('\n')

    print('done')
    print(done)
    print('\n')

    print('info')
    print(info)
    print('\n')