import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import gym
# for ddpg
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# for ppo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

distance_weight = 1.0
drift_weight = 0.00
shake_weight = 0.00
stand_weight = 1.0

class RobotEnv(gym.Env):
    def __init__(self):
        self.physicsClient = None
        self.robotID = None
        self.force = 100
        self.maxVelocity = 10
        self.startPos = [0., 0., .5]
        self.startOri = [np.pi/2, 0, 0]
        self.curPos = self.startPos
        self.lastPos = self.startPos
        self.curOri = self.startOri
        self.lastOri = self.startOri
        self.jointState = np.array([0.]*12)
        self.initial_action = []
        self.jointId = []
        self.distance_threshold = 0.01
        self.target_distance = 5.
        self.SLEEPTIME = 1/240
        self.n_sim_steps = np.int64(1 / self.SLEEPTIME * 5)
        self.step_n = 0
        self.jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]

        self._objectives = []
        self.objective_weight = [distance_weight, drift_weight, shake_weight, stand_weight]
        self.jointOffsets = []
        for i in range(4):
            self.jointOffsets.append(0)
            self.jointOffsets.append(-0.7)
            self.jointOffsets.append(0.7)

        ob_sp = np.array([np.pi]*12)
        self.observation_space = gym.spaces.Box(
            low=np.append(-ob_sp, np.array([0.,0,])),
            high=np.append(ob_sp, np.array([100., np.pi])),
            dtype=np.float32
        )
        '''
        self.action_space = gym.spaces.Box(
            low= np.array([-np.pi/100]*12),
            high=np.array([np.pi/100]*12),
            dtype=np.float32
        )
        '''
        self.action_space = gym.spaces.Box(
            low= np.array([-np.pi/100]*4),
            high=np.array([np.pi/100]*4),
            dtype=np.float32
        )
        #self.reset()

    # action: array[12]
    def applyAction2(self, action):
        idx = [1,2,7,8]
        for i in range(4):
            targetPos = float(action[i])
            targetPos+=self.jointState[idx[i]]
            p.setJointMotorControl2(bodyIndex=self.robotID,
                                    jointIndex=self.jointId[idx[i]],
                                    targetPosition=targetPos,
                                    controlMode=p.POSITION_CONTROL, #position/velocity
                                    force=self.force, #maxforce
                                    maxVelocity=self.maxVelocity)
            p.setJointMotorControl2(bodyIndex=self.robotID,
                                    jointIndex=self.jointId[idx[i]+3],
                                    targetPosition=targetPos,
                                    controlMode=p.POSITION_CONTROL, #position/velocity
                                    force=self.force, #maxforce
                                    maxVelocity=self.maxVelocity)

    def applyAction(self, action):
        #action = np.clip(action, -np.pi/3, np.pi/3)
        for j in range(0,12):
            targetPos = float(action[j])
            targetPos += self.jointState[j]
            p.setJointMotorControl2(bodyIndex=self.robotID,
                                    jointIndex=self.jointId[j],
                                    targetPosition=targetPos,
                                    controlMode=p.POSITION_CONTROL, #position/velocity
                                    force=self.force, #maxforce
                                    maxVelocity=self.maxVelocity)

    def setAngle(self, action=np.array([0.]*12)):
        for j in range(0,12):
            targetPos = action[j] * self.jointDirections[j] + self.jointOffsets[j]
            p.setJointMotorControl2(bodyIndex=self.robotID,
                                    jointIndex=self.jointId[j],
                                    targetPosition=targetPos,
                                    controlMode=p.POSITION_CONTROL, #position/velocity
                                    force=self.force, #maxforce
                                    maxVelocity=self.maxVelocity)

    def step(self, action):
        self.lastPos, self.lastOri = p.getBasePositionAndOrientation(self.robotID,
                                    physicsClientId=self.planeId)
        self.applyAction2(action)
        p.stepSimulation()
        obs = self.__get_observation()
        reward = self._reward()
        done = self._termination()
        self.step_n += 1
        if self.step_n > self.n_sim_steps:
            #print("n: ", self.step_n)
            done = True
        if done:
            self.step_n = 0
        info = {}
        return obs, reward, done, info


    def _reward(self):
        # reward for going forwards
        forward_reward = self.curPos[0] - self.lastPos[0]
        # Penalty for sideways translation.
        drift_reward = -abs(self.curPos[1] - self.lastPos[1])
        # Penalty for sideways rotation of the body.
        rot_mat = p.getMatrixFromQuaternion(self.curOri)
        local_up_vec = rot_mat[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # reward for being standing
        stand_reawrd = np.float32(self.curPos[2] > 0.27) - 1
        # add them up
        objectives = [forward_reward, drift_reward, shake_reward, stand_reawrd]
        weighted_objectives = [o * w for o, w in zip(objectives, self.objective_weight)]
        reward = sum(weighted_objectives)
        self._objectives.append(objectives)
        return reward

    def _termination(self):
        distance = np.sqrt(self.curPos[0]**2+self.curPos[1]**2)
        return distance>self.target_distance or self.is_fallen()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def __get_observation(self):
        self.curPos, self.curOri = p.getBasePositionAndOrientation(self.robotID,
                                    physicsClientId=self.planeId)
        obs = np.array([], dtype=np.float32)
        for i in range(12):
            self.jointState[i],_,_,_ = p.getJointState(self.robotID, i)
        obs = self.jointState.copy()
        distance = np.linalg.norm(np.array(self.curPos))
        obs = np.append(obs,distance)
        matrix = p.getMatrixFromQuaternion(self.curOri, physicsClientId=self.planeId)
        direction_vector = np.array([matrix[0], matrix[3], matrix[6]])
        position_vector = np.array(self.curPos)
        d_L2 = np.linalg.norm(direction_vector)
        p_L2 = np.linalg.norm(position_vector)
        if d_L2 == 0 or p_L2 == 0:
            obs = np.append(obs, np.pi)
        else:
            obs = np.append(obs, np.arccos(np.dot(direction_vector, position_vector)
                                  / (d_L2 * p_L2)))
        return obs.astype(np.float32)


    def reset(self):
        p.resetSimulation()
        p.setGravity(0,0,-10)

        self.curPos = self.startPos
        self.lastPos = self.startPos
        self.curOri = self.startOri
        self.lastOri = self.startOri
        # reset step_n
        self.step_n = 0
        # ref frame (id: 0)
        self.planeId = p.loadURDF("plane.urdf")
        # robot (id: 1)
        self.robotID = p.loadURDF("laikago/laikago.urdf", self.startPos,
                                 p.getQuaternionFromEuler(self.startOri))

        for j in range(p.getNumJoints(self.robotID)):
            #p.changeDynamics(self.robotID, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotID, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointId.append(j)
        self.setAngle()
        p.stepSimulation()
        self.lastPos, self.lastOri = p.getBasePositionAndOrientation(self.robotID,
                                    physicsClientId=self.planeId)
        return self.__get_observation()

    def is_fallen(self):
        rot_mat = p.getMatrixFromQuaternion(self.curOri)
        local_up = rot_mat[6:]
        #print("local_up: ", local_up)
        #print(np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)))
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) > 0.5 or
                self.curPos[2] < 0.13)

def trainDDPG():
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    env = RobotEnv()
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=1000, log_interval=1)
    model.save("ddpg")
    env = model.get_env()

    del model # remove to demonstrate saving and loading

def trainPPO(path="ppo"):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    env = RobotEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300000, log_interval=10)
    model.save(path)
    del model


if __name__ == "__main__":
    trainPPO("ppo_sep43")
