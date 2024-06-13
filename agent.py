import torch
import torch.nn as nn
import numpy as np
import tqdm

from model import *

class Agent():
    def __init__(self, env, model, device = 'cuda' if torch.cuda.is_available() else 'cpu', epsilon = 0.1,
                 gamma = 0.99, lr = 0.01, batch_size = 32, memory_size = 10000):
        self.env = env
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        self.criterion = nn.MSELoss()
        
        self.policy_net = model
        self.target_net = model
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.n_actions)
        else:
            with torch.no_grad():
                action =  self.policy_net(state).argmax().item()
                return action
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        try:
            action_batch = torch.cat(batch.action)
        except:
            action_tensors = [torch.tensor(action) for action in batch.action]
            print(action_tensors)
            action_batch = torch.cat(action_tensors)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self, num_episodes = 100, max_steps = 100):
        for episode in tqdm.tqdm(range(num_episodes), desc = 'Training', unit = 'episode', position = 0, leave = True):
            init_model, state, _ = self.env.reset()
            state = state.to(self.device)
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = torch.tensor(next_state, device = self.device, dtype = torch.float)
                reward = torch.tensor([reward], device = self.device, dtype = torch.float)
                
                self.memory.push(state, action, next_state, reward)
                
                state = next_state
                
                self.optimize_model()
                
                if done:
                    break
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f'Episode {episode + 1}, Reward = {reward.values()}, Observations = {self.env.get_observation()}')
            self.save('prunedmodel/pruned_model.pth', 'prunedmodel/dqn.pth')
        print('Training complete')
    
    def save(self, pruned_model_path, dqn_path):
        torch.save(self.env.model.state_dict(), pruned_model_path)
        torch.save(self.model.state_dict(), dqn_path)
    
    