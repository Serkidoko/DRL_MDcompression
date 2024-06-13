import numpy as np
import torch
import torchprofile
import random
from thop import profile
import time
import logging

logging.getLogger().setLevel(logging.WARNING)



def measure_flops(model,device, input_size = (1, 1, 784)):
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(inputs,))
    return flops

def calculate_acc(model, dataloader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


def calculate_sparsity(model):
    total = 0
    non_zero = 0
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight'):
            num_weights = layer.weight.numel()
            num_non_zero = layer.weight.detach().cpu().numpy().flatten().nonzero()[0].size
            total += num_weights
            non_zero += num_non_zero
    return 1 - non_zero / total

def calculate_inference_time(model, device = 'cuda', input_size = (1, 1, 784), iterations = 1):
    model.eval()
    model.to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            inputs = torch.randn(input_size).to(device)
            outputs = model(inputs)
    end_time = time.time()
    return (end_time - start_time) / iterations

def get_model_size(model):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # Convert bytes to megabytes (MB)
    
    return size_all_mb

def remove_random_weight(model):
    for name, param in model.named_parameters():
            weights = param.data
            num_weights = weights.numel() 
            random_index = random.randint(0, num_weights - 1)
            weights.view(-1)[random_index] = 0.0
            print(f"Removed weight at index {random_index}") 
            break
def get_total_weights(model):
    total = 0
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight'):
            num_weights = layer.weight.numel()
            total += num_weights
    return total