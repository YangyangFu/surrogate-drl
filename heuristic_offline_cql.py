import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from utils.replaybuffer import ReplayBuffer
import tqdm

import argparse
import gym
import gym_singlezone_jmodelica

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
    parser.add_argument('--lr', type=float, default=0.001)#0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)

    parser.add_argument('--epoch', type=int, default=100)#300

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



class DQN_h(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_h, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)
        

class Agent_h:
    def __init__(self, gamma, batch_size, device) -> None:
        self.env = Gym_Wrapper(make_building_env(args))
        self.device = device
        self.current_model = DQN_h(self.env.observation_space.shape[0] + 1, self.env.action_space.n).to(device)
        self.target_model  = DQN_h(self.env.observation_space.shape[0] + 1, self.env.action_space.n).to(device)
        self.update_target()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr = args.lr)
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.current_model.forward(state)
            action  = q_value.max(1)[1].data[0]
            action  = action.detach().cpu().item()#.numpy()
        else:
            action = random.randint(0, self.env.action_space.n-1)
        return action

    def collect(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def Q_function(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.current_model.forward(state)
        return q_value[0]

    
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())
    
    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state) 

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        cql1_loss = torch.logsumexp(q_values, dim=1).mean() - q_values.mean()
        
        loss = cql1_loss + 0.5*F.mse_loss(q_value, expected_q_value.data)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            loss = self.compute_td_loss(self.batch_size)
            return loss.data
        return None

    def save(self, model_path):
        torch.save(self.current_model.state_dict(), model_path + "offline_cql_current_model.pt")
        torch.save(self.target_model.state_dict(), model_path + "offline_cql_target_model.pt")

    def load(self, model_path):
        self.current_model.load_state_dict(torch.load(model_path + "offline_cql_current_model.pt"))
        self.target_model.load_state_dict(torch.load(model_path + "offline_cql_target_model.pt"))

    def load_replaybuffer(self):
        file_path="./history_data/"
        state_list = np.load(file_path+'state_list.npy')
        action_list = np.load(file_path+'action_list.npy')
        reward_list = np.load(file_path+'reward_list.npy')
        next_state_list = np.load(file_path+'next_state_list.npy')
        done_list = np.load(file_path+'done_list.npy')

        len_ = len(state_list)
        for i in range(len_):
            self.collect(state_list[i], action_list[i], reward_list[i], next_state_list[i], done_list[i])


class heuristic:
    def __init__(self, device, printable = False, load_agent = False) -> None:
        
        #device = torch.device('cuda')
        batch_size = args.batch_size
        gamma      = args.gamma
        
        self.env = Gym_Wrapper(make_building_env(args))
        self.DQNAgent = Agent_h(gamma, batch_size, device)
        self.printable = printable
        if load_agent:
            self.DQNAgent.load(model_path = "./models/")
        else:
            self.build_heuristic_1()
            self.DQNAgent.save(model_path = "./models/")
        
        

    def build_heuristic_1(self, num_epochs = 100):

        self.DQNAgent.load_replaybuffer()

        epsilon_start = args.eps_train
        epsilon_final = args.eps_train_final
        epsilon_decay = args.epoch * args.step_per_epoch / 2.0 #to be determined

        epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
        losses = []
        all_rewards = []
        episode_reward = 0
        
        
        state = self.env.reset()
        for ep in tqdm.tqdm(range(args.epoch)):
            for frame_idx in range(args.step_per_epoch):
                #epsilon = epsilon_by_frame(frame_idx)
                
                #action = self.DQNAgent.act(state, epsilon)
                #next_state, reward, done, _ = self.env.step(action)
                
                #self.DQNAgent.collect(state, action, reward, next_state, done)
                
                #state = next_state
                #episode_reward += reward
                
                
                    
                loss1 = self.DQNAgent.learn()
                if loss1: losses.append(loss1)

                if frame_idx % 100 == 0: self.DQNAgent.update_target()
                    
                #if done:
                #    state = self.env.reset()
                #    if self.printable: print(episode_reward)
                #    all_rewards.append(episode_reward)
                #    episode_reward = 0
                #    break
            
            

    def get_v(self, state):
        q_value = self.DQNAgent.Q_function(state).detach().cpu().numpy()
        #print(q_value)
        #q_value = q_value.data
        n_actions = self.env.action_space.n
        v, v_ind = np.max(q_value), np.argmax(q_value)
        return v, v_ind

    def get_v_with_constraints(self, state, pre_mean, pre_std, accepted_action_set):
        state = np.array(state).reshape(1,-1)*pre_std + pre_mean

        q_value = self.DQNAgent.Q_function(state).detach().cpu().numpy()
        #print(q_value)
        #q_value = q_value.data
        #n_actions = self.env.action_space.n

        if len(accepted_action_set) == 0:
            v, v_ind = np.max(q_value), np.argmax(q_value)
            return v, v_ind

        q_value_accepted = q_value[0][accepted_action_set]
        v = np.max(q_value_accepted)
        v_ind = 0

        for i in range(len(q_value_accepted)):
            if abs(q_value_accepted[i] - v) < 1e4:
                v_ind = accepted_action_set[i]
        return v, v_ind

    def test_avg_acc_reward(self, test_rounds = 4, max_iters = 300):
        epsilon = 0.05
        
        
        all_rewards = []
        episode_reward = 0
        
        state = self.env.reset()
        for ep in range(test_rounds):
            for iters in range(max_iters):
                action = self.DQNAgent.act(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                
                if done:
                    state = self.env.reset()
                    all_rewards.append(episode_reward)
                    episode_reward = 0
        return np.mean(all_rewards)
    
from new_env.read_npy import plot_one_ep_normalized

env = make_building_env(args)
env = Gym_Wrapper(env)
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
        #reward_list.append(reward)
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
    return plot_one_ep_normalized(num_zone = 1, history_Z_T = l_indoor, history_Env_T = l_outdoor, his_cost = c, num_days = args.num_of_days, fig_path_name = './heu_offline_cql_test_data_simulation.png')

if __name__ == "__main__":
    device = torch.device('cuda')



    heu = heuristic(train_env=False, device = device, printable=True, load_agent = False)
    
    test_agent(agent = heu.DQNAgent)

    avg_reward = heu.test_avg_acc_reward()
    print("average accumulated reward from heuristic: ", avg_reward)

    env = Gym_Wrapper(make_building_env(args))
    env.seed(11)
    state = env.reset()
    print("state", state, ", get_v: ", heu.get_v(state)) 



        

