import torch
import torch.nn as nn
import numpy as np
import tqdm
from torch.distributions import Categorical
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
        state = state.float().unsqueeze(0)
        probs = self.policy_net(state)
        m = Categorical(probs)  
        action = m.sample()
        return action.item(), m.log_prob(action)
        
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
            init_model, state, log_prob = self.env.reset()
            state = torch.tensor(state, device = self.device, dtype = torch.float)
            log_probs = []
            rewards = []
            for step in range(max_steps):
                action, current_log_prob = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = torch.tensor(next_state, device = self.device, dtype = torch.float)
                reward = torch.tensor([reward], device = self.device, dtype = torch.float)  
                # log_probs = torch.tensor([log_probs], device = self.device, dtype = torch.float)
            
                log_probs.append(current_log_prob)
                rewards.append(reward)

                
                # self.memory.push(state, action, next_state, reward)
                
                state = next_state
            
                self.optimize_model()
                
                if done:
                    break

            rewards_np = np.array([r.item() for r in rewards[::-1]])  
            log_probs_np = np.array([lp.item() for lp in log_probs]) 
            cumulative_reward = 0
            G = [] 
            for r in rewards_np:
                cumulative_reward = r + self.gamma * cumulative_reward
                G.insert(0, cumulative_reward)
            
            G = torch.tensor(G, device=self.device, dtype=torch.float)
            G = (G - G.mean()) / (G.std() + 1e-9)

            policy_loss = []
            for log_prob, G in zip(log_probs_np, G):
                policy_loss.append(-log_prob * G)
            policy_loss = torch.stack(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f'Episode {episode + 1}, Reward = {reward.values()}, Observations = {self.env.get_observation()}')
            self.save('prunedmodel/pruned_model.pth', 'prunedmodel/dqn.pth')
        print('Training complete')
    
    def save(self, pruned_model_path, dqn_path):
        torch.save(self.env.model.state_dict(), pruned_model_path)
        torch.save(self.model.state_dict(), dqn_path)
    
    