
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
import random
from utils.replaybuffer import ReplayBuffer

import math
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
        
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x= None, train_y=None, likelihood=None, mean_module = None, base_covar_module = None, covar_module = None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.base_covar_module = base_covar_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class Q_GP():
    def __init__(self, input_dim, output_dim, lr, device):
        self.lr = lr
        self.device = device
        self.input_dim = input_dim
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.n_inducing = 200
        self.inducing_points = torch.randn(self.n_inducing, input_dim).to(device)

        self.mean_module = ConstantMean().to(device)
        self.base_covar_module = ScaleKernel(RBFKernel()).to(device)
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=self.inducing_points, likelihood=self.likelihood).to(device)
        X = torch.FloatTensor(np.float32(np.random.normal(size=(1, input_dim)))).to(self.device)
        y = torch.FloatTensor(np.float32([10])).to(self.device)
        self.learn(X, y, num_steps = 0)


        

    def learn(self, X, y, num_steps = 1):
        self.model = GPRegressionModel(train_x = X, train_y = y, likelihood = self.likelihood, mean_module = self.mean_module, base_covar_module = self.base_covar_module, covar_module = self.covar_module).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(num_steps):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

    def predict(self, X):
        self.model.eval()
        with gpytorch.settings.max_preconditioner_size(100), torch.no_grad():
            preds = self.model(X)
        loc, var = preds.mean, preds.variance
        return loc, var

class Q_GP_multi():
    def __init__(self, input_dim, output_dim, lr, device):
        self.n = output_dim
        self.lr = lr
        self.device = device
        self.model = [Q_GP(input_dim, output_dim, lr, device) for i in range(output_dim)]
    
    def learn(self, X, Y, ind, num_steps=1):
        for i in range(self.n):
            selected = (ind == i)
            if len(Y[selected]) > 0:
                self.model[i].learn(X[selected], Y[selected], num_steps=num_steps)
        return None
        

    def predict(self, X):
        y = []
        y_var = []
        for i in range(self.n):
            loc, var = self.model[i].predict(X)
            y.append(torch.unsqueeze(loc, dim = 1))
            y_var.append(torch.unsqueeze(var, dim = 1))
        Y = torch.cat(y, dim=1)
        Y_var = torch.cat(y_var, dim=1)
        return Y, Y_var

    def set_param(self,Q_GP_multi1):
        for i in range(self.n):
            if self.model[i].model:
                self.model[i].model.load_state_dict(Q_GP_multi1.model[i].model.state_dict())
    

class Agent:
    def __init__(self, env, gamma, batch_size, ReplayBuffer_size, device) -> None:
        self.env = env
        self.device = device
        self.current_model = Q_GP_multi(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, lr=0.003, device=device)
        self.target_model  = Q_GP_multi(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, lr=0.003, device=device)
        self.update_target()
        self.replay_buffer = ReplayBuffer(ReplayBuffer_size)
        self.gamma_initial = gamma
        self.gamma = gamma
        self.batch_size = batch_size
        self.l_step = 1

        self.lambda_value = 1.0
        


    def update_lambda_gamma(self, iters, total_iters):
        self.lambda_value = 1.0 - (iters/total_iters)
        if self.lambda_value < 0: self.lambda_value = 0
        self.gamma = self.gamma_initial * self.lambda_value
        return

    def act(self, state, epsilon):
        if random.random() > epsilon and self.current_model.model[0].model:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value, q_var = self.current_model.predict(state)
            #print(q_value, q_var, len(self.replay_buffer))
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
        #self.target_model.sgpr.parameters()
        self.target_model.set_param(self.current_model)
    
    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample_for_GP()

        state      = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)

        q_values, q_var      = self.current_model.predict(state)
        next_q_values, next_q_var = self.current_model.predict(next_state)
        next_q_state_values, next_q_s_var = self.target_model.predict(next_state) 

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        self.current_model.learn(state, expected_q_value, torch.max(next_q_values, 1)[1], num_steps=1)
        #loss = (q_value - expected_q_value.data).pow(2).mean()
            
        #self.optimizer.zero_grad()
        #loss.backward()
        #self.optimizer.step()
        
        return
    
    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            for i in range(self.l_step):
                self.compute_td_loss(self.batch_size)
        return None
        



if __name__ == "__main__":
    
    N = 1000
    X = torch.distributions.uniform.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    y = 1000 * torch.sin(torch.cat([torch.unsqueeze(torch.mean(X, dim = 1),dim=1), torch.unsqueeze(torch.mean(X, dim = 1)*2, dim=1)], dim=1)) + torch.distributions.normal.Normal(0.0, 0.2).sample(sample_shape=(N,2))

    X1 = torch.distributions.uniform.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    y1 = torch.sin(torch.cat([torch.unsqueeze(torch.mean(X, dim = 1),dim=1), torch.unsqueeze(torch.mean(X, dim = 1)*2, dim=1)], dim=1)) * 1000+ torch.distributions.normal.Normal(0.0, 0.2).sample(sample_shape=(N,2))
    
    device = torch.device('cpu')
    
    current_model = Q_GP_multi(10, 2, 0.01, device)
    target_model  = Q_GP_multi(10, 2, 0.01, device)
    

    #current_model.learn(X, y, num_steps = 0)
    #target_model.learn(X, y, num_steps = 0)

    y_1, var_1 = current_model.predict(X)
    print(torch.mean(y_1-y1))

    y_2, var_2 = target_model.predict(X)
    print(torch.mean(y_2-y1))
    
    #target_model.sgpr.load_state_dict(current_model.sgpr.state_dict())
    

    current_model.learn(X, y, num_steps = 100)
    

    
    y_1, var_1 = current_model.predict(X)
    print(torch.mean(y_1-y1))

    y_2, var_2 = target_model.predict(X)
    print(torch.mean(y_2-y1))









