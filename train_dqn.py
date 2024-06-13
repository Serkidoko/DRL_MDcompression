from agent import *
from env import *
from utils import *
from model import PolicyNetwork, Model

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import logging
logging.getLogger().setLevel(logging.WARNING)



train_loader = torch.utils.data.DataLoader(MNIST('data', train = True, download = False, transform = ToTensor()), batch_size = 64, shuffle = True)

model = Model(784, 128, 10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)
env = PruningEnv(model, train_loader, device, layers_name=['l1', 'l2'])
n_actions = env.n_actions
n_observations = env.n_observations
# DQN_model = DQN(n_observations, n_actions).to(device)
# agent = Agent(env, DQN_model, device)

policy_model = PolicyNetwork(n_observations, n_actions).to(device)
agent = Agent(env, policy_model, device)


agent.train()
agent.save(pruned_model_path='pruned_model_last.pth', dqn_path='dqn_last.pth')