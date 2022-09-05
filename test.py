#from kyzh_env import *
from laikago_dof4 import *

from stable_baselines3.common.env_checker import check_env

def checkEnv():
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    env = RobotEnv()
    check_env(env)

def showDDPG():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    env = RobotEnv()
    model = DDPG.load("ddpg")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

def showPPO(path="ppo"):
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    env = RobotEnv()
    model = PPO.load(path)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    showPPO("ppo_sep42")
