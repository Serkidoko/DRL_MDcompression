import torch
import torch.nn as nn   
import numpy as np
from copy import deepcopy
from utils import *


class PruningEnv(object):
    def __init__(self, model, dataloader, device, layers_name, ckpt_path = 'model.ckpt'):
       self.model = model
       self.dataloader = dataloader
       self.init_model = deepcopy(model)
       self.device = device
       self.layers_name = layers_name
       self.ckpt_path = ckpt_path
       self.n_actions = self.__get_action_space()
       self.n_observations = len(self.get_state())
       self.observations = self.__get_observation()
       
       self.reset()

    def reset(self):
        self.model = deepcopy(self.init_model)
        info = self.__get_observation()
        state = self.get_state()
        return self.model, state, info   
    
    def get_observation(self):
        return self.__get_observation()
        
    def __get_observation(self):
        return {
            'flops': measure_flops(self.model, self.device),
            'acc': calculate_acc(self.model, self.dataloader, self.device),
            'sparsity': calculate_sparsity(self.model),
            'inference_time': calculate_inference_time(self.model),
            'size': get_model_size(self.model)
        }
    
    def __calculate_reward(self, current_observation, new_observation):
        reward = 0
        if current_observation['flops'] > new_observation[0]:
            reward += 1
        else:
            reward -= 1
        if current_observation['acc'] - new_observation[1] < 0.5:
            reward += 1
        else:
            reward -= 1
        if current_observation['sparsity'] < new_observation[2]:
            reward += 1
        else:
            reward -= 1
        if current_observation['inference_time'] > new_observation[3]:
            reward += 1
        else:
            reward -= 1
        if current_observation['size'] > new_observation[4]:
            reward += 1
        else:
            reward -= 1
        return reward
    
    def __terminate(self, current_observation, new_observation):
        '''
        Acc:  0.9581
        Infer time:  0.0
        Sparsity:  0.0
        FLOPs:  101632.0
        Model size:  0.38823699951171875
        '''
        if current_observation['acc'] - new_observation[1] < 0.:
            return True
        if current_observation['inference_time'] < new_observation[3]:
            return True
        if current_observation['size'] >= new_observation[4]:
            return True
        return False



    def get_state(self):
        obs = self.__get_observation().values()
        return torch.tensor(list(obs)).to(self.device)
        
    
    def step(self, action):
        if action == 0:
            return self.get_state(), \
                self.__calculate_reward(self.__get_observation(), self.get_state()), \
                self.__terminate(self.__get_observation(), self.get_state())
        elif action == 1:
            remove_random_weight(self.model)
            return self.get_state(), \
                self.__calculate_reward(self.__get_observation(), self.get_state()), \
                self.__terminate(self.__get_observation(), self.get_state())
                    
        
    
    def render(self):
        pass
    
    def __get_action_space(self):
        return 2
    
    def __get_observation_space(self):
        return get_total_weights(self.model)