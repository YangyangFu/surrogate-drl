import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils.plot import plot, plot_single, plot_array
from utils.agent import Agent

from heuristic_surrogate_env import heuristic as heuristic_s
#from lib.TD3_BC.main import heuristic as heuristic_o
from heuristic_offline_cql import heuristic as heuristic_o


from engineered_guidance import get_engineered_guidance

import collections

import argparse
import gym
import gym_singlezone_jmodelica

import tqdm

from new_env.read_npy import plot_one_ep_normalized


use_init = True
use_heu_surrogate = True
use_heu_offline = True
use_engineered_guidance = True
eval_ep = 2
alp = 0.004
initial_lambda = 0.5
lr_setting = 0.001
n_trials_setting = 4

#reward res = (penalty + cost*50.0) / 10.0

def get_args(folder="experiment_results"):
    time_step = 15*60.0
    num_of_days = 2#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSSingleZoneEnv-v1")
    parser.add_argument('--nActions', type=int, default=51)
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num-of-days', type=int, default=num_of_days)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0003)#0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)

    parser.add_argument('--epoch', type=int, default=120)#300

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
        cost, penalty = cost[0], penalty[0]
        
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
num_actions = args.nActions
action_space = 1
mean = 0.0
std = 1.0

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = args.epoch * args.step_per_epoch / 1.0 #to be determined

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


batch_size = args.batch_size
gamma      = args.gamma
ReplayBuffer_size = args.buffer_size
target_update_interval = args.target_update_freq


def test_agent_sub(file_path = "./history_data/", agent = None, sub_fig = '1'):

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
    return plot_one_ep_normalized(num_zone = 1, history_Z_T = l_indoor, history_Env_T = l_outdoor, his_cost = c, num_days = args.num_of_days, fig_path_name = './HVAC_fig/DDQN_'+sub_fig+'.png')

def run_experient(trial=1):
    losses = []
    all_rewards = []
    episode_reward = 0
    avg_reward = collections.deque(maxlen = 100)

    DQNAgent = Agent(env.state_dim, env.action_n, gamma, batch_size, ReplayBuffer_size, lr_setting, device)

    list_voilation_rate, list_voilation_val_mean, list_total_cost = [], [], []
    
    voilation_rate, voilation_val_mean, total_cost = test_agent_sub(file_path = "./history_data/", agent = DQNAgent, sub_fig=str(trial)+"ep0")
    list_voilation_rate.append(voilation_rate)
    list_voilation_val_mean.append(voilation_val_mean)
    list_total_cost.append(total_cost)

    state = env.reset()
    

    for ep in tqdm.tqdm(range(args.epoch)):
        for frame_idx in range(args.step_per_epoch):
            epsilon = epsilon_by_frame(frame_idx)
            
            action = DQNAgent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            DQNAgent.collect(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            loss1 = DQNAgent.learn()
            if loss1: losses.append(loss1)
                
            if frame_idx % target_update_interval == 0:
                DQNAgent.update_target()

            if done:
                
                state = env.reset()
                print(episode_reward)
                all_rewards.append(episode_reward)
                episode_reward = 0
                break
        
        if (ep+1) % eval_ep == 0:
            voilation_rate, voilation_val_mean, total_cost = test_agent_sub(file_path = "./history_data/", agent = DQNAgent, sub_fig=str(trial)+"ep"+str(ep+1))
            list_voilation_rate.append(voilation_rate)
            list_voilation_val_mean.append(voilation_val_mean)
            list_total_cost.append(total_cost)
            state = env.reset()
        
    
    

    return all_rewards, [list_voilation_rate, list_voilation_val_mean, list_total_cost]


def engineered_guidance_engaged(heuristic_list, state, env_name):


    accepted_action_set = get_engineered_guidance(state, num_actions, env_name, normalized_state = True)

    ans = None
    for heu in heuristic_list:
        v_h, v_ind = heu.get_v_with_constraints(state, mean, std, accepted_action_set)
        if ans == None:
            ans = v_h
        else:
            ans = min(ans, v_h)
    return ans

def generate_dataset_with_q_value(heuristic_list, history_data):
    X = history_data
    Y = []
    

    for x in history_data:
        arr = []
        for h in heuristic_list:
            arr.append(h.DQNAgent.Q_function(x).detach().cpu().numpy())
        arr = np.asarray(arr)
        y_ = np.min(arr, axis = 0)
        #print(x, y_)

        Y.append(y_)
    return X, Y

def run_experient_heuristic(trial = 1, set_heuristic = True, train_heuristic = False):
    def test_avg_acc_reward(test_rounds = 3, max_iters = 300):
        epsilon = 0.05
        
        
        all_rewards = []
        episode_reward = 0
        
        state = env.reset()
        for ep in range(test_rounds):
            for frame_idx in range(args.step_per_epoch):
                    action = DQNAgent.act(state, epsilon)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        state = env.reset()
                        all_rewards.append(episode_reward)
                        episode_reward = 0
                        break
        return np.mean(all_rewards)

    losses = []
    all_rewards = []
    episode_reward = 0
    all_rewards_o = []
    episode_reward_o = 0

    DQNAgent = Agent(env.state_dim, env.action_n, gamma, batch_size, ReplayBuffer_size, lr_setting, device)

    DQNAgent.lambda_value = initial_lambda

    

    heuristic_list = []

    if use_heu_offline:
        heu1 = heuristic_o(device = device, printable=True, load_agent = not train_heuristic)
        heuristic_list.append(heu1)
    
    if use_heu_surrogate:
        heu2 = heuristic_s(train_env=False, device = device, printable=True, load_agent = not train_heuristic)
        heuristic_list.append(heu2)


    #########initialize from heuristics############
    if use_init:
        file_path = "./history_data/"
        state_list = np.load(file_path+'state_list.npy')
        X, Y = generate_dataset_with_q_value(heuristic_list, state_list)
        DQNAgent.initialize_agent_from_dataset(X, Y, nepochs = 30, lr = 0.01)
    ###############################################


    list_voilation_rate, list_voilation_val_mean, list_total_cost = [], [], []
    
    
    voilation_rate, voilation_val_mean, total_cost = test_agent_sub(file_path = "./history_data/", agent = DQNAgent, sub_fig=str(trial)+"ep0")
    list_voilation_rate.append(voilation_rate)
    list_voilation_val_mean.append(voilation_val_mean)
    list_total_cost.append(total_cost)
    

    state = env.reset()

    acc_frame_idx = 0
    for ep in tqdm.tqdm(range(args.epoch)):
        for frame_idx in range(args.step_per_epoch):
            acc_frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)
            
            action = DQNAgent.act(state, epsilon)
            next_state, reward_o, done, _ = env.step(action)
            #print(DQNAgent.gamma)
            ##########heuristic guided##########################
            if use_heu_surrogate or use_heu_offline:
                #v_h_1, v_ind_1 = heu1.get_v(state, mean, std)
                #v_h_2, v_ind_2 = heu2.get_v(state)
                #v_h = min(v_h_1, v_h_2)
                if use_engineered_guidance:
                    v_h = engineered_guidance_engaged(heuristic_list, next_state, env_id)
                else:
                    if use_heu_offline:
                        v_h_1, v_ind_1 = heu1.get_v(next_state)
                        v_h = v_h_1
                    if use_heu_surrogate:
                        v_h_2, v_ind_2 = heu2.get_v(next_state)
                        if use_heu_offline:
                            v_h = min(v_h_1, v_h_2)
                        else:
                            v_h = v_h_2
                    
                #print(v_h, reward_o)
                reward = reward_o + (1-DQNAgent.lambda_value)*DQNAgent.gamma*v_h
                #(args.step_per_epoch*args.epoch)//lambda_affected_len # to be determined 2
                DQNAgent.update_lambda_gamma(acc_frame_idx, initial_lambda = initial_lambda, alp = alp)
            else:
                reward = reward_o
            ####################################################
            
            DQNAgent.collect(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_reward_o += reward_o
            
            
                
            loss1 = DQNAgent.learn()
            if loss1: losses.append(loss1)
                
            if frame_idx % target_update_interval == 0:
                DQNAgent.update_target()

            if done:
                state = env.reset()
                print(episode_reward, episode_reward_o)
                all_rewards.append(episode_reward)
                all_rewards_o.append(episode_reward_o)
                
                episode_reward = 0
                episode_reward_o = 0
                break

        if (ep+1) % eval_ep == 0:
            voilation_rate, voilation_val_mean, total_cost = test_agent_sub(file_path = "./history_data/", agent = DQNAgent, sub_fig=str(trial)+"ep"+str(ep+1))
            list_voilation_rate.append(voilation_rate)
            list_voilation_val_mean.append(voilation_val_mean)
            list_total_cost.append(total_cost)
            state = env.reset()
        
    #test_avg_acc_reward()

    return all_rewards_o, [list_voilation_rate, list_voilation_val_mean, list_total_cost]




if __name__ == "__main__":
    

    train_mode = 1
    file_path = "./figures/"
    if train_mode == 0: #DDQN
        n_trials = 4
        result = []
        x = []
        y = []
        min_len = 1000000000

        x_list_voilation_rate, y_list_voilation_rate = [], []
        x_list_voilation_val_mean, y_list_voilation_val_mean = [], []
        x_list_total_cost, y_list_total_cost = [], []
        x_axis = [int(i*eval_ep) for i in range(1+(args.epoch)//eval_ep)]

        for i in range(n_trials):
            t, his_lists = run_experient(trial=i)

            x.extend(list(range(len(t))))
            y.extend(t)
            min_len = min(min_len, len(t))

            list_voilation_rate, list_voilation_val_mean, list_total_cost = his_lists
            x_list_voilation_rate.extend(x_axis)
            y_list_voilation_rate.extend(list_voilation_rate)
            x_list_voilation_val_mean.extend(x_axis)
            y_list_voilation_val_mean.extend(list_voilation_val_mean)
            x_list_total_cost.extend(x_axis)
            y_list_total_cost.extend(list_total_cost)

        plot_array([[np.asarray(x_list_voilation_rate),np.asarray(y_list_voilation_rate)]], (len(x_axis)-1)*eval_ep, file_path+'violation_rate_DDQN.png', x_name='Training Episodes', y_name='Voilation Rate')
        plot_array([[np.asarray(x_list_voilation_val_mean),np.asarray(y_list_voilation_val_mean)]], (len(x_axis)-1)*eval_ep, file_path+'violation_value_DDQN.png', x_name='Training Episodes', y_name='Voilation Value')
        plot_array([[np.asarray(x_list_total_cost),np.asarray(y_list_total_cost)]], (len(x_axis)-1)*eval_ep, file_path+'cost_DDQN.png', x_name='Training Episodes', y_name='Cost')


        
        plot_single([[np.asarray(x),np.asarray(y)]], min_len, file_path+'reward_DDQN.png')


        
    elif train_mode == 1: # heuristic guided DDQN

        train_heuristic = False

        fig_name = ''
        if use_heu_offline:
            fig_name += 'offline_'
        if use_heu_surrogate:
            fig_name += 'surrogate_'
        if use_engineered_guidance:
            fig_name += 'eng_'
        if use_init:
            fig_name += 'init_'
        

        if train_heuristic:
            n_trials = 1
        else:
            n_trials = n_trials_setting
        result = []
        x = []
        y = []
        min_len = 1000000000

        x_list_voilation_rate, y_list_voilation_rate = [], []
        x_list_voilation_val_mean, y_list_voilation_val_mean = [], []
        x_list_total_cost, y_list_total_cost = [], []
        x_axis = [i*eval_ep for i in range(1+(args.epoch)//eval_ep)]


        for i in range(n_trials):
            t, his_lists = run_experient_heuristic(trial = i, set_heuristic=use_heu_offline or use_heu_surrogate, train_heuristic = train_heuristic)
            x.extend(list(range(len(t))))
            y.extend(t)
            min_len = min(min_len, len(t))

            list_voilation_rate, list_voilation_val_mean, list_total_cost = his_lists
            x_list_voilation_rate.extend(x_axis)
            y_list_voilation_rate.extend(list_voilation_rate)
            x_list_voilation_val_mean.extend(x_axis)
            y_list_voilation_val_mean.extend(list_voilation_val_mean)
            x_list_total_cost.extend(x_axis)
            y_list_total_cost.extend(list_total_cost)
        
        
        plot_array([[np.asarray(x_list_voilation_rate),np.asarray(y_list_voilation_rate)]], (len(x_axis)-1)*eval_ep, file_path+'violation_rate_DDQN_'+fig_name+'png', x_name='Training Episodes', y_name='Voilation Rate')
        plot_array([[np.asarray(x_list_voilation_val_mean),np.asarray(y_list_voilation_val_mean)]], (len(x_axis)-1)*eval_ep, file_path+'violation_value_DDQN_'+fig_name+'png', x_name='Training Episodes', y_name='Voilation Value')
        plot_array([[np.asarray(x_list_total_cost),np.asarray(y_list_total_cost)]], (len(x_axis)-1)*eval_ep, file_path+'cost_DDQN_'+fig_name+'png', x_name='Training Episodes', y_name='Cost')
        
        plot_single([[np.asarray(x),np.asarray(y)]], min_len, file_path+'reward_Hu_DDQN_'+fig_name+'.png')
        
        
    

    
        

