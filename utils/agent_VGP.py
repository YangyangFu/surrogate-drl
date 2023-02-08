
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
        
print_q_value = False

class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents = 3, num_tasks = 4, n_inducing = 20, input_dim = 10):
        # Let's use a different set of inducing points for each task
        inducing_points = torch.rand(num_tasks, n_inducing, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents = 3, num_tasks = 4, n_inducing = 20, input_dim = 10):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, n_inducing, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Q_VGP():
    def __init__(self, input_dim, output_dim, lr, device):
        self.lr = lr
        self.device = device
        self.model = MultitaskGPModel(num_latents = 10, num_tasks = output_dim, n_inducing = 200, input_dim = input_dim).to(device)#pre num_latents = 10, n_inducing = 200
        #self.model = IndependentMultitaskGPModel(num_latents = 20, num_tasks = output_dim, n_inducing = 300, input_dim = input_dim).to(device)#pre num_latents = 10, n_inducing = 200
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(device)
        self.singlelikelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.model.parameters()},
                        {'params': self.likelihood.parameters()},
                        {'params': self.singlelikelihood.parameters()},
                    ], lr=lr)

    def learn_test(self, X, Y, num_steps=500):
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=Y.size(0))

        for i in range(num_steps):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, Y)
            loss.backward()
            self.optimizer.step()
        

    def predict(self, X):
        self.model.eval()
        preds = self.model(X)
        loc, var = preds.mean, preds.variance

        return loc, var

    def set_param(self, model1):
        self.model.load_state_dict(model1.model.state_dict())
    

class Agent:
    def __init__(self, env, gamma, batch_size, ReplayBuffer_size, device) -> None:
        self.env = env
        self.device = device
        self.current_model = Q_VGP(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, lr=0.003, device=device)#0.003 0.001
        self.target_model  = Q_VGP(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, lr=0.003, device=device)
        self.update_target()
        self.replay_buffer = ReplayBuffer(ReplayBuffer_size)

        self.gamma_final = gamma
        self.gamma = gamma
        self.batch_size = batch_size

        self.gamma_initial = 0.01
        self.lambda_value = 1.0
        self.calls = 0

        self.l_step = 1
        


    def update_lambda_gamma(self, iters, total_iters):
        f = lambda frame_idx: 1.0 + (0.01 - 1.0) * math.exp(-1. * frame_idx / 100)
        self.lambda_value = f(float(iters))
        #if self.lambda_mul_gamma > 0.8: self.lambda_mul_gamma = 0.99
        self.gamma = self.gamma_final * self.lambda_value
        return

    def act(self, state, epsilon):
        self.calls += 1
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value, q_var = self.current_model.predict(state)
            if print_q_value and self.calls % 30 == 0:
                print(q_value, q_var, len(self.replay_buffer))
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
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)

        self.current_model.model.train()

        

        #q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        #next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        #expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        '''
        q_value = self.current_model.model(state, task_indices=action)
        with torch.no_grad():
            next_q_value = self.target_model.model(next_state, task_indices=torch.max(next_q_values, 1)[1]).mean
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        '''
        
        '''
        for i in range(30):
            with torch.no_grad():
                next_q_values = self.current_model.model(next_state).mean
                q_value = self.current_model.model(state)
                max_action = torch.max(next_q_values, 1)[1]
                next_q_value = self.target_model.model(next_state).mean
                target = q_value.mean
                
                for i in range(len(action)):
                    target[i, action[i]] = reward[i] + self.gamma * next_q_value[i, max_action[i]] * (1 - done[i])

            
            q_value = self.current_model.model(state)
            mll = gpytorch.mlls.VariationalELBO(self.current_model.likelihood, self.current_model.model, num_data=state.size(0))
            #print(target.shape)
            loss = -mll(q_value, target)
            #print("loss: ", loss.item())
                
            self.current_model.optimizer.zero_grad()
            loss.backward()
            self.current_model.optimizer.step()
        '''

        for i in range(1):
            with torch.no_grad():
                next_q_values = self.current_model.model(next_state).mean
                next_q_value = self.target_model.model(next_state, task_indices=torch.max(next_q_values, 1)[1]).mean
                expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            
            q_value = self.current_model.model(state, task_indices=action)
            mll = gpytorch.mlls.VariationalELBO(self.current_model.singlelikelihood, self.current_model.model, num_data=state.size(0))
            #print(target.shape)
            loss = -mll(q_value, expected_q_value)
            #print("loss: ", loss.item())
                
            self.current_model.optimizer.zero_grad()
            loss.backward()
            self.current_model.optimizer.step()
        
        return
    
    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            for i in range(self.l_step):
                self.compute_td_loss(self.batch_size)
        return None
        



if __name__ == "__main__":
    
    N = 1600
    X = torch.distributions.uniform.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    y = 1000 * torch.sin(torch.cat([torch.unsqueeze(torch.mean(X, dim = 1),dim=1), torch.unsqueeze(torch.mean(X, dim = 1)*2, dim=1)], dim=1)) + torch.distributions.normal.Normal(0.0, 0.2).sample(sample_shape=(N,2))

    X1 = torch.distributions.uniform.Uniform(0.0, 1.0).sample(sample_shape=(N,10))
    y1 = torch.sin(torch.cat([torch.unsqueeze(torch.mean(X, dim = 1),dim=1), torch.unsqueeze(torch.mean(X, dim = 1)*2, dim=1)], dim=1)) * 1000+ torch.distributions.normal.Normal(0.0, 0.2).sample(sample_shape=(N,2))
    
    device = torch.device('cpu')
    
    current_model = Q_VGP(10, 2, 0.1, device)
    target_model  = Q_VGP(10, 2, 0.1, device)
    

    #current_model.learn(X, y, num_steps = 0)
    #target_model.learn(X, y, num_steps = 0)

    y_1, var_1 = current_model.predict(X)
    print(y_1.shape)
    print(torch.mean(y_1-y1))

    y_2, var_2 = target_model.predict(X)
    print(torch.mean(y_2-y1))
    
    #target_model.sgpr.load_state_dict(current_model.sgpr.state_dict())
    

    #for i in range(20):
    #    current_model.learn_test(X[i*50:(i+1)*50], y[i*50:(i+1)*50], num_steps = 10)
    current_model.learn_test(X, y, num_steps = 10)

    
    y_1, var_1 = current_model.predict(X1)
    print(torch.mean(y_1-y1))

    y_2, var_2 = target_model.predict(X1)
    print(torch.mean(y_2-y1))









