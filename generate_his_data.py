import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import gym
import gym_singlezone_jmodelica


from new_env.read_npy import plot_one_ep_normalized

from utils.agent import Agent
from utils.replaybuffer import ReplayBuffer
from lib.TD3_BC.TD3_BC_utils import ReplayBuffer as history_collector
import collections

import tqdm

test_agent_only = False

def get_args(folder="experiment_results"):
    time_step = 15*60.0
    num_of_days = 2#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSSingleZoneEnv-v1")
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num-of-days', type=int, default=num_of_days)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.001)#0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)

    parser.add_argument('--epoch', type=int, default=150)#200

    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)

    parser.add_argument('--logdir', type=str, default='log')
    
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=folder)

    parser.add_argument('--test-only', type=bool, default=False)


    return parser.parse_args()
args = get_args()

def make_building_env(args):
    weather_file_path = "./USA_CA_Riverside.Muni.AP.722869_TMY3.epw"
    mass_flow_nor = [0.75]
    npre_step = 3
    simulation_start_time = 212*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 7
    alpha = 1
    nActions = 51

    def rw_func(cost, penalty):

        if ( not hasattr(rw_func,'x')  ):
            rw_func.x = 0
            rw_func.y = 0

        cost, penalty = cost[0], penalty[0]

        if rw_func.x > cost:
            rw_func.x = cost
        if rw_func.y > penalty:
            rw_func.y = penalty

        print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)

        
        res = (penalty + cost*100.0)

        return res

    env = gym.make(args.task,
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   npre_step = npre_step,
                   simulation_start_time = simulation_start_time,
                   simulation_end_time = simulation_end_time,
                   time_step = args.time_step,
                   log_level = log_level,
                   alpha = alpha,
                   nActions = nActions,
                   rf = rw_func)
    return env


class Gym_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super(Gym_Wrapper, self).__init__(env)
        self.h = np.array([86400., 273.15+30, 273.15+40,1200., 1000.]+[273.15+40]*3+[1200.]*3 + [args.num_of_days*24*3600.0])
        self.l = np.array([0., 273.15+12, 273.15+0,0, 0]+[273.15+0]*3+[0.0]*3 + [0.0])
        self.time_cnt = 0.0
        self.state_dim = self.env.observation_space.shape[0] + 1
        self.action_n = self.env.action_space.n
    
    def step(self, action):
        ob, r, d, info = self.env.step(action)

        self.time_cnt += args.time_step
        ob = np.append(ob, [self.time_cnt])

        
        ob = (ob - self.l)/(self.h-self.l)
        #print(ob)
        return ob, r, d, info

    def reset(self):
        self.time_cnt = 0.0
        ob = self.env.reset()
        ob = np.append(ob, [self.time_cnt])
        ob = (ob - self.l)/(self.h-self.l)
        return ob


    def get_cost(self):
        return self.env.get_cost()


device = torch.device('cuda')


env_id = args.task
env = make_building_env(args)
env = Gym_Wrapper(env)

epsilon_start = args.eps_train
epsilon_final = args.eps_train_final
epsilon_decay = args.epoch * args.step_per_epoch / 1.0 #to be determined

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


batch_size = args.batch_size
gamma      = args.gamma
ReplayBuffer_size = args.buffer_size
target_update_interval = args.target_update_freq

def test_agent(file_path = "./history_data/", agent = None):

    state_list, action_list, reward_list, cost_list, next_state_list = [], [], [], [], []


    state = env.reset()
    for frame_idx in range(args.step_per_epoch):
        epsilon = 0.05
        
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        sub_cost = env.get_cost()

        state_list.append(state)
        action_list.append(action)
        reward_list.append(reward)
        cost_list.append(sub_cost)
        next_state_list.append(next_state)
        
        state = next_state

    #np.save(file_path + '/main_his_sta.npy', state_list)
    #np.save(file_path + '/main_his_act.npy', action_list)
    #np.save(file_path + '/main_his_reward.npy', reward_list)
    #np.save(file_path + '/main_his_cost.npy', cost_list)
    #np.save(file_path + '/main_his_next_sta.npy', next_state_list)

    #l = np.load(file_path + '/main_his_sta.npy', allow_pickle=True)
    l = np.asarray(state_list)
    l_indoor = l[:, 1]
    l_outdoor = l[:, 2]
    #c = np.load(file_path + '/main_his_cost.npy', allow_pickle=True)
    c = np.asarray(cost_list)
    return plot_one_ep_normalized(num_zone = 1, history_Z_T = l_indoor, history_Env_T = l_outdoor, his_cost = c, num_days = args.num_of_days, fig_path_name = './generate_data_simulation.png')

        

    

def run_experient(file_path = "./history_data/", num_test_epochs = 10, epsilon_test = 0.05):
    losses = []
    all_rewards = []
    episode_reward = 0

    DQNAgent = Agent(env.state_dim, env.action_n, gamma, batch_size, ReplayBuffer_size, device)

    state = env.reset()

    for ep in tqdm.tqdm(range(args.epoch)):
        for frame_idx in range(args.step_per_epoch):
            epsilon = epsilon_by_frame(frame_idx)
            
            action = DQNAgent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            DQNAgent.collect(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = env.reset()
                print(episode_reward)

                all_rewards.append(episode_reward)

                
                episode_reward = 0
                
            loss1 = DQNAgent.learn()
            if loss1: losses.append(loss1)
                
                
            if frame_idx % target_update_interval == 0:
                DQNAgent.update_target()
    

    test_agent(file_path, DQNAgent)

    if not test_agent_only:

        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []

        state = env.reset()
        for ep in range(num_test_epochs):
            for frame_idx in range(args.step_per_epoch):
                epsilon = epsilon_test
                
                action = DQNAgent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)

                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                next_state_list.append(next_state)
                done_list.append(done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    state = env.reset()
                    print(episode_reward)
                    episode_reward = 0

        state_list = np.asarray(state_list)
        np.save(file_path+'state_list.npy', state_list)
        action_list = np.asarray(action_list)
        np.save(file_path+'action_list.npy', action_list)
        reward_list = np.asarray(reward_list)
        np.save(file_path+'reward_list.npy', reward_list)
        next_state_list = np.asarray(next_state_list)
        np.save(file_path+'next_state_list.npy', next_state_list)
        done_list = np.asarray(done_list)
        np.save(file_path+'done_list.npy', done_list)


    return

def load_history_data(state_dim, action_dim, file_path):
    state_list = np.load(file_path+'state_list.npy')
    action_list = np.load(file_path+'action_list.npy')
    reward_list = np.load(file_path+'reward_list.npy')
    next_state_list = np.load(file_path+'next_state_list.npy')
    done_list = np.load(file_path+'done_list.npy')

    replay_buffer = history_collector(state_dim, action_dim)
    replay_buffer.state = state_list
    replay_buffer.action = action_list.reshape(-1,action_dim)
    replay_buffer.next_state = next_state_list
    replay_buffer.reward = reward_list.reshape(-1,1)
    replay_buffer.not_done = 1. - done_list.reshape(-1,1)
    replay_buffer.size = replay_buffer.state.shape[0]

    return replay_buffer
    


if __name__ == "__main__":
    file_path = "./history_data/"
    run_experient(file_path = file_path, num_test_epochs = 30, epsilon_test = 0.05)

    


    

        
        
    

    
        

