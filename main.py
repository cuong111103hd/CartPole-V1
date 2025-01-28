import gymnasium as gym #them thu vien moi truong
import numpy as np #them thu vien tinh toan
import time

#Create environment
env = gym.make('CartPole-v1') #Specify

np.random.seed(1)
HighReward = 0
BestWeights = None


for i in range(200):
    observation, info = env.reset()
    Weights = np.random.uniform(-1,1,4)
    Sumreward = 0
    for j in range(1000):
        env.render()
        action = 0 if np.matmul(Weights, observation) < 0 else 1
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample()) #Tra ve cac bien cho moi lan step
        Sumreward += reward
        print(i,j,Weights,observation,action,Sumreward,BestWeights)

    if Sumreward > HighReward:
        HighReward = Sumreward
        BestWeights = Weights



print(env.spec)
