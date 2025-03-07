import gym
import time

env = gym.make('CartPole-v0')

observation = env.reset()

for i in range(1000):
    time.sleep(0.01)
    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    if pole_ang > 0:
        action = 1
    else:
        action = 0

    observation, reward, done, info = env.step(action)

# if done:
# 	break

env.close()
