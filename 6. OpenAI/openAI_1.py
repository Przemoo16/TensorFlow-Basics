import gym

env = gym.make('CartPole-v0')

print("INITIAL OBSERVATION")
observation = env.reset()
print(observation)

for i in range(2):
    
    #env.render()

    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)

    print("Performed One Random Action")
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

env.close()
