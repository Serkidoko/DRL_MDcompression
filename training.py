import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.nn as nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return F.softmax(out, dim=1)
    

transforms = transforms.Compose([transforms.ToTensor()])
    
train_loader = torch.utils.data.DataLoader(MNIST('./data', train=True, transform = transforms,download=True), batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(MNIST('./data', train=False, transform = transforms), batch_size=100, shuffle=False)

model = Model(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
        
        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {i + 1}, Loss = {loss.item()}, Accuracy = {accuracy.item()}')
            
torch.save(model.state_dict(), 'model.ckpt')