import gym

env = gym.make('CartPole-v0')

observation = env.reset()

for t in range(200):

    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    if pole_ang > 0: # if pole tilts to right

        action = 1 # slide wagon to right
    
    else: # if pole tilts to left

        action = 0

    observation, reward, done, info = env.step(action)

