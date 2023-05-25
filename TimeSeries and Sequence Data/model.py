import torch 
import torch.nn as nn 

# N = num of samples
# T = sequence len
# D = num of input features ( D > 1 e.g multiple brainwave scans in parallel)
# M = num of hidden units
# K = num of output units (K>1, lat longitudinal coords regression e.g )

device = torch.device('cuda:0')
# Many to one RNN
# RNN model requires shape N x T x 1 (batch_first else T x N x D)
class SimpleRNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_layers, num_outputs):
        super().__init__()
        self.D = num_inputs
        self.M = num_hidden
        self.L =  num_layers
        self.K = num_outputs

        # Number of samples x sequence length x number of features
        self.rnn =nn.RNN(
            input_size = self.D,
            hidden_size = self.M,
            num_layers = self.L,
            nonlinearity = 'relu',
            batch_first = True,
        )

        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        #initial hidden states - L x N x M, number of layers x batch_size x number hidden features
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        # size (N, T, M)
        out, _ = self.rnn(X, h0)

        # we only want h(T) at the final step, 
        # output size = N x M -> N x K (output node)
        out = self.fc(out[:, -1, :])

        return out
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()

    def forward(self, X):
        pass