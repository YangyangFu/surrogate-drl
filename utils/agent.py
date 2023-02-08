
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from utils.replaybuffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
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
        

class Agent:
    def __init__(self, input_dim, output_dim, gamma, batch_size, ReplayBuffer_size, lr, device) -> None:
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_model = DQN(input_dim, output_dim).to(device)
        self.target_model  = DQN(input_dim, output_dim).to(device)
        self.update_target()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr = lr)#0.001
        self.replay_buffer = ReplayBuffer(ReplayBuffer_size)
        
        self.gamma_final = gamma
        self.gamma = gamma
        self.batch_size = batch_size

        self.lambda_value = None
        
    def initialize_agent_from_dataset(self, data_x, data_y, nepochs = 10, lr = 0.1):

        print("initialize_agent_from_dataset, the size of the dataset = ", len(data_x))
        batch_size = 64
        nepoch = nepochs
        n = len(data_x)
        n_iters = n // batch_size
        optimizer = torch.optim.Adam(self.current_model.parameters(), lr = lr)
        criterion = torch.nn.MSELoss()
        cnt = 0

        for epoch in range(nepoch):
            for i in range(n_iters):
                st = i*batch_size
                ed = min((i+1)*batch_size, n-1)
                inputs, labels = torch.FloatTensor(data_x[st:ed]).to(self.device), torch.FloatTensor(data_y[st:ed]).to(self.device)
                labels = labels.float()

                out = self.current_model(inputs)
                
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cnt += 1
                if i % 20 == 0:
                    print('Epoch:{0},Frame:{1}, mse_loss {2}'.format(epoch, cnt*batch_size, loss))


    def update_lambda_gamma(self, iters, initial_lambda = 1.0, alp = 0.01):
        f = lambda frame_idx: initial_lambda + (1.0 - initial_lambda) * math.tanh(alp*frame_idx)#math.exp(-1. * frame_idx / total_iters)
        self.lambda_value = f(float(iters))
        self.gamma = self.gamma_final * self.lambda_value
        return

    def act(self, state, epsilon):
        if random.random() > epsilon and self.current_model:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.current_model.forward(state)
            #if self.calls % 30 == 0:
            #    print(q_value, len(self.replay_buffer))
            action  = q_value.max(1)[1].data[0]
            action  = action.detach().cpu().item()#.numpy()
        else:
            action = random.randint(0, self.output_dim-1)
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
        
        loss = (q_value - expected_q_value.data).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            loss = self.compute_td_loss(self.batch_size)
            return loss.data
        return None
        














