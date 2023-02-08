from json import load
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import argparse
import gym
import gym_singlezone_jmodelica

use_state_normalization = True

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

    parser.add_argument('--epoch', type=int, default=300)

    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)

    parser.add_argument('--logdir', type=str, default='log')
    
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=folder)

    parser.add_argument('--test-only', type=bool, default=False)


    return parser.parse_args()

args = get_args()

def make_building_env(args = get_args()):
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


class Gym_Wrapper_HVAC(gym.Wrapper):
    def __init__(self, device, trainable = False, his_data_path = "./history_data/", model_path = './models/HVAC_env_linear.pt'):
        self.args = get_args()
        self.env = Gym_Wrapper(make_building_env(self.args))
        super(Gym_Wrapper_HVAC, self).__init__(self.env)
        
        self.state = self.env.reset()
        
        self.trans = Transition_wo_reward(state_dim = self.env.state_dim, action_dim = 1, device = device).to(device)

        self.steps = 0

        self.device = device
        self.model_path = model_path
        self.his_data_path = his_data_path

        if trainable:
            self.train_surrogate_model(n_ep = 160)
        else:
            self.load_surrogate_model()

        

        

    def train_surrogate_model(self, n_ep = 100):#40
        def load_history_data(action_dim, file_path):
            state_list = np.load(file_path+'state_list.npy')
            action_list = np.load(file_path+'action_list.npy')
            next_state_list = np.load(file_path+'next_state_list.npy')

            state = state_list
            action = action_list.reshape(-1,action_dim)
            next_state = next_state_list
            return state, action, next_state
        
        data_state, data_action, data_next_state = load_history_data(1, self.his_data_path)

        #we only need indoor temperature and power as label
        ind = np.asarray([1,4])#temperature, power
        data_next_state = data_next_state[:, ind]

        data_len = len(data_state)
        print("historical data length: ", data_len)
        test_data_len = min(data_len // 10, 2000)
        train_data_len = data_len - test_data_len

        ind = np.arange(data_len)
        ind = np.random.permutation(ind)
        data_state, data_action, data_next_state = data_state[ind], data_action[ind], data_next_state[ind]

        test_state, test_action, test_next_state = data_state[:test_data_len], data_action[:test_data_len], data_next_state[:test_data_len]
        test_state, test_action, test_next_state = torch.FloatTensor(test_state).to(self.device), torch.FloatTensor(test_action.astype(float)).to(self.device), torch.FloatTensor(test_next_state).to(self.device)

        data_state, data_action, data_next_state = data_state[test_data_len:], data_action[test_data_len:], data_next_state[test_data_len:]


        batch_size = 32
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.trans.parameters(), lr=0.0001, weight_decay=1e-5)#

        cnt = 0

        
        for epoch in range(n_ep):

            for i in range(int(train_data_len/batch_size)):
                self.trans.train()

                be = (i)*batch_size
                ed = (i+1)*batch_size
                if ed >= train_data_len: ed = train_data_len - 1

                state = torch.FloatTensor(data_state[be: ed])
                action = torch.FloatTensor(data_action[be: ed].astype(float))
                next_state = torch.FloatTensor(data_next_state[be: ed])

                state, action, next_state = state.to(self.device), action.to(self.device), next_state.to(self.device)
                
                out = self.trans.forward(state, action)

                loss = criterion(out, next_state)
                #loss = criterion(out, label.squeeze())
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cnt += 1
                if i % 1000 == 0:
                    self.trans.eval()
                    out = self.trans.forward(test_state, test_action)

                    #print(out[0], test_next_state[0])

                    out_, test_next_state_ = out*18.0+12.0, test_next_state*18.0+12.0
                    loss = torch.sqrt(criterion(out_, test_next_state_))

                    print('Epoch:{0},Frame:{1}, abs diff {2}'.format(epoch, cnt*batch_size, loss.item()))
            
        torch.save(self.trans.state_dict(), self.model_path)

    def load_surrogate_model(self):
        self.trans.load_state_dict(torch.load(self.model_path))

        
    def if_done(self, state):
        done = bool(
            self.steps >= self.args.step_per_epoch

        )


        return done
    
    def step(self, action):

        #other variables in states should be the same as self.env
        real_ob, _, d, _  = self.env.step(action)

        pred_temp, pred_pow = self.trans.forward_single(self.state, action)
        ob = real_ob
        ob[1] = pred_temp
        ob[4] = pred_pow

        self.state = ob
        self.steps += 1
        #d = self.if_done(ob)



        states = ob
        if use_state_normalization:
            power = states[4] * 1000.0
            time = int(states[0] * 86400.0)
            TZon = (states[1] * 18 + 273.15+12) - 273.15
        else:
            power = states[4]
            time = states[0]
            TZon = states[1] - 273.15 # orginal units from Modelica are SI units
        
        # Here is how the reward should be calculated based on observations
        
        num_zone = 1
        ZTemperature = [TZon] #temperature in C for each zone
        ZPower = [power] # power in W
        
        T_upper = [30.0 for i in range(24)]
        T_lower = [12.0 for i in range(24)]
        T_upper[7:19] = [26.0]*12
        T_lower[7:19] = [22.0]*12
        
        # control period:
        delCtrl = self.env.tau/3600.0 #may be better to set a variable in initial
        
        #get hour index
        t = int(time)
        t = int((t%86400)/3600) # hour index 0~23

        #calculate penalty for each zone
        overshoot = []
        undershoot = []
        max_violation = []
        penalty = [] #temperature violation for each zone
        cost = [] # erengy cost for each zone

        for k in range(num_zone):
            overshoot.append(max(ZTemperature[k] - T_upper[t] , 0.0))
            undershoot.append(max(T_lower[t] - ZTemperature[k] , 0.0))
            max_violation.append(-overshoot[k] - undershoot[k])
            penalty.append(self.alpha*max_violation[k])
        
        t_pre = int(time-self.tau) if time>self.tau else (time+24*60*60.-self.tau)
        t_pre = int((t_pre%86400)/3600) # hour index 0~23
        
        for k in range(num_zone):
            cost.append(- ZPower[k]/1000. * delCtrl * self.p_g[t_pre])
        
        # save cost/penalty for customized use - negative
        self._cost = cost
        self._max_temperature_violation = max_violation

        # define reward
        if self.rf:
            rewards=self.rf(cost, penalty)
            #print(rewards)
        else:
            rewards=np.sum(np.array([cost, penalty]))

        r = rewards

        #if r < -6: r = -6.0

        #print(cost, penalty)

        return ob, r, d, None

    def reset(self, seed = None):
        
        self.state = self.env.reset()
        self.steps = 0

        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    


class Transition_wo_reward(nn.Module):
    def __init__(self, state_dim, action_dim, device, pred_dim = 2):
        super(Transition_wo_reward, self).__init__()
        self.device = device

        '''
        self.layers1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), #nn.ReLU(),
            nn.Linear(256, 256), 
        )
        self.layers2 = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.layer0 = nn.Linear(256, 256)

        self.layers3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        '''
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), #nn.ReLU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 256), 
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 256), 
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, pred_dim)
        )
        
        

    def forward(self, state, action):
        NN_input = torch.cat([state, action], 1)

        #a1 = self.layers1(NN_input)
        #a2 = self.layers2(a1)
        #a2 = torch.cat([self.layer0(a1), a2], 1)
        #a3 = self.layers3(a2)
        #a = a3
        
        a = self.layers(NN_input)
        return a

    def forward_single(self, state_, action_):
        state = torch.FloatTensor(state_).to(self.device)
        state = state.unsqueeze(0)
        action = torch.FloatTensor(np.asarray([float(action_)])).to(self.device)
        action = action.unsqueeze(0)
        #print(state, action)
        NN_input = torch.cat([state, action], 1)
        a = self.layers(NN_input)
        a = a[0].detach().cpu().numpy()
        return a[0], a[1]



if __name__ == "__main__":
    device = torch.device('cuda')
    env = Gym_Wrapper_HVAC(device, trainable=True)#, his_data_path = "../history_data/", model_path = '../models/HVAC_env_linear.pt'