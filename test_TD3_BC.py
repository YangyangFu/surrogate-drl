import numpy as np
import torch
import argparse
import os
import math

from lib.TD3_BC import TD3_BC
import sys

import gym
import gym_singlezone_jmodelica

from lib.TD3_BC.main import heuristic

#reward res = (penalty + cost*50.0) / 10.0

def get_args(folder="experiment_results"):
    time_step = 15*60.0
    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSSingleZoneEnv-v1")
    parser.add_argument('--nActions', type=int, default=51)
    
    num_of_days = 2#31
    parser.add_argument('--num-of-days', type=int, default=num_of_days)
    
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)
    
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.001)#0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)

    parser.add_argument('--epoch', type=int, default=200)

    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)

    parser.add_argument('--logdir', type=str, default='log')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=folder)

    parser.add_argument('--test-only', type=bool, default=False)


    return parser.parse_args()

HVAC_args = get_args()

def make_building_env(args = get_args()):
    weather_file_path = "./USA_CA_Riverside.Muni.AP.722869_TMY3.epw"
    mass_flow_nor = [0.75]
    n_next_steps = 3
    simulation_start_time = 212*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 7
    alpha = 1
    nActions = args.nActions

    def rw_func(cost, penalty):
        cost, penalty = cost[0], penalty[0]
        res = (penalty + cost*100.0)
        return res

    env = gym.make(args.task,
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   n_next_steps = n_next_steps,
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
        self.h = np.array([86400., 273.15+30, 273.15+40,1200., 1000.]+[273.15+40]*3+[1200.]*3 + [HVAC_args.num_of_days*24*3600.0])
        self.l = np.array([0., 273.15+12, 273.15+0,0, 0]+[273.15+0]*3+[0.0]*3 + [0.0])
        self.time_cnt = 0.0
        self.state_dim = self.env.observation_space.shape[0] + 1
        self.action_n = self.env.action_space.n
    
    def step(self, action):
        ob, r, d, info = self.env.step(action)

        self.time_cnt += HVAC_args.time_step
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
        
from new_env.read_npy import plot_one_ep_normalized

def test_heu_agent(file_path = "./history_data/", heu = None):

    state_list, action_list, reward_list, cost_list, next_state_list = [], [], [], [], []


    state = env.reset()
    done = False
    while(not done):
        epsilon = 0.05
        state_normd = (np.array(state).reshape(1,-1) - heu.mean)/heu.std
        action = heu.agent.select_action(state_normd)

        action = max(action[0], 0)
        action = round(action)
        action = min(action, HVAC_args.nActions-1)

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
    return plot_one_ep_normalized(num_zone = 1, history_Z_T = l_indoor, history_Env_T = l_outdoor, his_cost = c, num_days = HVAC_args.num_of_days, fig_path_name = './heu_offline_test_data_simulation.png')
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Gym_Wrapper(make_building_env(HVAC_args))



    heu = heuristic(env, env_name=HVAC_args.task, device = device, train_model = True, num_actions = env.action_space.n, action_space = 1)
    
    
    test_heu_agent(heu = heu)
    
    env.seed(11)
    state = env.reset()
    print("state", state, ", get_v: ", heu.get_v(state, 0, 1)) 

    