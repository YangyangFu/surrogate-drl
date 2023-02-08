
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
import random
from utils.replaybuffer import ReplayBuffer

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from blitz.modules.base_bayesian_module import BayesianModule
        
@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianRegressor, self).__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 128, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = True)
        self.blinear2 = BayesianLinear(128, 128, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = True)
        self.blinear3 = BayesianLinear(128, output_dim, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)
        
    def forward(self, t):
        
                
        if isinstance(t, tuple):
            x, action = t[0], t[1]
            x = F.relu(self.blinear1(x))
            x = F.relu(self.blinear2(x))
            x = self.blinear3(x)
            q_values      = x
            q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
            return q_value
        else:
            x = t
            x = F.relu(self.blinear1(x))
            x = F.relu(self.blinear2(x))
            return self.blinear3(x)

class Q_BNN():
    def __init__(self, input_dim, output_dim, lr, device):
        self.device = device
        self.Net = BayesianRegressor(input_dim=input_dim, output_dim=output_dim).to(device)
        
        self.Net.unfreeze_()

        self.optimizer = optim.Adam(self.Net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def learn(self, X, action, y, sample_nbr=10, n_updates = 4):
        for i in range(n_updates):
            self.optimizer.zero_grad()
            loss = self.Net.sample_elbo(inputs=(X, action), labels=y,
                            criterion=self.criterion,
                            sample_nbr=sample_nbr,
                            complexity_cost_weight=1/X.shape[0])
            
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), 0.5)
            self.optimizer.step()

    def predict(self, X, n_samples = 20):
        pred = [self.Net(X) for i in range(n_samples)]
        preds = torch.stack(pred)
        loc, var = preds.mean(axis=0), preds.std(axis=0)
        return loc, var
    

class Agent:
    def __init__(self, env, gamma, batch_size, ReplayBuffer_size, device) -> None:
        self.env = env
        self.device = device
        self.current_model = Q_BNN(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, lr=0.003, device=device)
        self.target_model  = Q_BNN(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, lr=0.003, device=device)
        self.update_target()
        self.replay_buffer = ReplayBuffer(ReplayBuffer_size)
        self.gamma_initial = gamma
        self.gamma = gamma
        self.batch_size = batch_size

        self.lambda_value = 1.0

        self.count = 0
        


    def update_lambda_gamma(self, iters, total_iters):
        self.lambda_value = 1.0 - (iters/total_iters)
        if self.lambda_value < 0: self.lambda_value = 0
        self.gamma = self.gamma_initial * self.lambda_value
        return

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value, q_var = self.current_model.predict(state)
            #print(q_value)
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
            q_value, q_var = self.current_model.predict(state)
        return q_value[0]

    
    def update_target(self):
        self.target_model.Net.load_state_dict(self.current_model.Net.state_dict())
    
    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size=batch_size)

        state      = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            q_values, q_var      = self.current_model.predict(state)
            next_q_values, next_q_var = self.current_model.predict(next_state)
            next_q_state_values, next_q_s_var = self.target_model.predict(next_state) 

            q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        #print(next_q_value)
        
        self.count += 1
        #if self.count > 1000:
        #    self.current_model.Net.unfreeze_()
        #    print("unfreeze")
            
        self.current_model.learn(state, action, expected_q_value)
        #loss = (q_value - expected_q_value.data).pow(2).mean()
            
        #self.optimizer.zero_grad()
        #loss.backward()
        #self.optimizer.step()
        
        return
    
    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            loss = self.compute_td_loss(self.batch_size)
            #return loss.data
        return None
        




if __name__ == "__main__":
    
    N = 1500
    X = torch.distributions.uniform.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    y = 1000 * torch.sin(3*torch.mean(X, dim = 1)) + torch.distributions.normal.Normal(0.0, 0.2).sample(sample_shape=(N,1))

    X1 = torch.distributions.uniform.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    y1 = 1000 * torch.sin(3*torch.mean(X1, dim = 1)) + torch.distributions.normal.Normal(0.0, 0.2).sample(sample_shape=(N,1))
    
    device = torch.device('cpu')
    current_model = Q_BNN(input_dim = 10, output_dim = 1, lr = 0.003, device = device)
    target_model  = Q_BNN(input_dim = 10, output_dim = 1, lr = 0.003, device = device)

    y_1, var_1 = current_model.predict(X, 100)
    print(torch.mean(y_1-y1), torch.mean(var_1))

    y_2, var_2 = target_model.predict(X, 100)
    print(torch.mean(y_2-y1), torch.mean(var_2))
    for j in range(10):
        for i in range(20):
        
            current_model.learn(X[i*64:(i+1)*64], y[i*64:(i+1)*64])

        y_1, var_1 = current_model.predict(X)
        print(torch.mean(y_1-y1), torch.mean(var_1))
    #target_model.learn(X, y)
    #target_model.Net.load_state_dict(current_model.Net.state_dict())
    

    #current_model.learn(X, y)
    

    
    y_1, var_1 = current_model.predict(X)
    print(torch.mean(y_1-y1), torch.mean(var_1))

    y_2, var_2 = target_model.predict(X)
    print(torch.mean(y_2-y1), torch.mean(var_2))


    '''
    pyro.clear_param_store()
    target_model  = Q_GP(10, 1)
    target_model.learn(X, y, num_steps = 0)

    y_2, var_2 = target_model.predict(X)
    print(torch.mean(y_2-y1))
    

    N = 1000
    X = dist.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    l1 = torch.unsqueeze(torch.sin(3*torch.mean(X, dim = 1)), dim=1)
    l2 = torch.unsqueeze(torch.cos(3*torch.mean(X, dim = 1)), dim = 1)
    lst = [l1, l2]
    y = 1000 * torch.cat(lst, dim = 1) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,2))
    
    current_model = Q_GP_multi(10, 2)
    target_model  = Q_GP_multi(10, 2)
    
    current_model.learn(X, y, num_steps = 10)
    target_model.learn(X, y, num_steps = 0)
   
    X = dist.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    l1 = torch.unsqueeze(torch.sin(3*torch.mean(X, dim = 1)), dim=1)
    l2 = torch.unsqueeze(torch.cos(3*torch.mean(X, dim = 1)), dim = 1)
    lst = [l1, l2]
    y = 1000 * torch.cat(lst, dim = 1) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,2))
    y_1, var_1 = current_model.predict(X)
    print(torch.mean(y_1-y))

    #target_model.sgpr.load_state_dict(current_model.sgpr.state_dict())
    y_2, var_2 = target_model.predict(X)
    print(torch.mean(y_2-y))
    '''






