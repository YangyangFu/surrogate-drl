import math, random

import gym
import numpy as np

import torch
import collections

def get_engineered_guidance(state, num_actions, env_name, normalized_state = False):
    #return accepted actions
    #aciton 0 ~ num_actions-1
    action_list = []
    if env_name == "CartPole-v0":
        Position, Velocity, Angle, Angular_Velocity = state
        
        for action in range(num_actions):
            if (action-0.5) * Angle > 0:
                action_list += [action]

        return action_list

    if env_name == "JModelicaCSSingleZoneEnv-v1":
        t = state[0]
        l_indoor = state[1]
        l_outdoor = state[2]

        if normalized_state:
            t = int(t * 86400)
            l_indoor = l_indoor * (30 - 12) + 12
            l_outdoor = l_outdoor * 40 + 0

        if 7*3600 < t < 19*3600: # in occupancy hour
            if l_indoor < 22:
                action_list += [0, 1, 2, 3]
            
            if l_indoor > 27:
                action_list += [num_actions-4, num_actions-3, num_actions-2, num_actions-1]
        else:
            action_list = [0, 1, 2] # otherwise, low power mode
        return action_list


        

    print("engineered guidance for the given env not implemented")

