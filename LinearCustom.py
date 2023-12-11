import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class LinearCustom(nn.Module):
    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, inputs, parameters):
        # print(inputs.shape, parameters[0].shape, parameters[1].shape)
        weights = parameters[0]
        biases = parameters[1]
        return torch.bmm(inputs, weights).squeeze(1) + biases


class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, input_dim, output_dim, num_nodes, dynamic=True):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic

        if self.dynamic:
            self.weight_generator = nn.Sequential(
                *[
                    nn.Linear(memory_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, 5),
                    nn.ReLU(),
                    nn.Linear(5, input_dim * output_dim),
                ]
            )
            self.bias_generator = nn.Sequential(
                *[
                    nn.Linear(memory_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, 5),
                    nn.ReLU(),
                    nn.Linear(5, output_dim),
                ]
            )
        else:
            print("Using FC")
            self.weights = nn.Parameter(
                torch.rand(input_dim, output_dim), requires_grad=True
            )
            self.biases = nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def forward(self, memory):
        if self.dynamic:
            weights = self.weight_generator(memory).view(
                self.num_nodes, self.input_dim, self.output_dim
            )
            biases = self.bias_generator(memory).view(self.num_nodes, self.output_dim)
        else:
            weights = self.weights
            biases = self.biases

        parameters = [weights, biases]
        return parameters
