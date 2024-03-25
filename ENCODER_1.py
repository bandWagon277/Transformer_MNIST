import math
import torch
import numpy as np
from torch import nn


# Define hyperparameters

n = 28      #input matrix: batch_size * n * n
d = 28      # dimension of data feature
k = 12      # dimension of W feature
h = 8       # number of heads
p = 100       # dimension of feed forward
L = 6       #number of layers

contact = False     #whether use block
m = 4       #block size: m * m
col = True      #Default vertical contact
overlap = False     #Default non-overlap
stride = 2      # stride length if overlap

pos = False     # whether use position emb



class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        all_pos = np.array([
            [pos / np.power(10000, 2 * i / d) for i in range(d)]
            if pos != 0 else np.zeros(d) for pos in range(max_len)])
        all_pos[1:, 0::2] = np.sin(all_pos[1:, 0::2])           # even number len
        all_pos[1:, 1::2] = np.cos(all_pos[1:, 1::2])           # odd number len
        self.pos_table = torch.FloatTensor(all_pos)        

    def forward(self, X):                              # X: [seq_len, d_model]    
        X += self.pos_table[:X.size(1), :]            # X: [batch_size, seq_len, d_model]
        return self.dropout(X)

# Compute attension score
class Attentionscore(nn.Module):
    def __init__(self):
        super(Attentionscore, self).__init__()

    def forward(self, Q, K, V):     # Q,K,V: 4d tensor with size: batch_size * h * n * k
        QK = torch.matmul(Q,K.transpose(-1, -2))/math.sqrt(k)
        alpha = nn.Softmax(dim=-1)(QK)
        alpha_V = torch.matmul(alpha,V)
        return alpha_V      #alpha_V: 4d tensor with size: batch_size * h * n * d

# Define function of layernorm
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))        #gamma,beta: 1d tensor with length d
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True)
        eps=1e-6
        return self.gamma * (x - mean) / (sd + eps) + self.beta
    

# compute Q,K,V and u
class selfattention(nn.Module):
    def __init__(self):
        super(selfattention, self).__init__()
        self.W_Q = nn.Linear(d, k * h, bias=False)
        self.W_K = nn.Linear(d, k * h, bias=False)
        self.W_V = nn.Linear(d, k * h, bias=False)
        self.W_c = nn.Linear(h * k, d, bias=False)

        
    def forward(self, X):    # X: 3d input data with size: batch_size * n * d
        batch_size = X.size(0)
        #print(X.size())
        Q = self.W_Q(X).view(batch_size, -1, h, k).transpose(1,2)  # Q,K,V: 4d tensor with size: batch_size * h * n * k
        K = self.W_K(X).view(batch_size, -1, h, k).transpose(1,2)  
        V = self.W_V(X).view(batch_size, -1, h, k).transpose(1,2)  
        alpha_V = Attentionscore()(Q, K, V)          #alpha_V: batch_size * h * n * d
        alpha_V = alpha_V.transpose(1, 2).reshape(batch_size, -1, h * k) # batch_size * n * h(d)
        u_0 = self.W_c(alpha_V)                                               # batch_size * n * d
        #return LayerNorm().cuda()(u_0 + X)
        return LayerNorm()(u_0 + X)

# feed forward conntetcion layer
class feedForward(nn.Module):
    def __init__(self):
        super(feedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d, p, bias=False),   #W1
            nn.ReLU(),
            nn.Linear(p, d, bias=False))   #w2

    def forward(self, u):                                  # u: batch_size * n * d
        z = self.ff(u)
        #return nn.LayerNorm(self.d).cuda()(z + u) 
        return LayerNorm()(z + u) 

# Model of encoder
class Encoderlayer(nn.Module):
    def __init__(self):
        super(Encoderlayer, self).__init__()
        self.selfAttention = selfattention() 
        self.feedForward = feedForward() 

    def forward(self, X):           # X: input data with size batch_size * n* d
        #batch_size = X.size(0)
        u0 = self.selfAttention(X)            
        u = self.feedForward(u0)
        #u=u.view(batch_size,28,28)
        return u,u0

# multiple layer of encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d)
        self.layers = nn.ModuleList([Encoderlayer() for _ in range(L)])

    def forward(self, X):
        output = self.pos_emb(X) if pos else X
        #output = X                                        
        for layer in self.layers:
            temp = layer(output)
            output,u = temp

        return output,u


# Require square matrix n*n

class ContactEncoder(nn.Module):
    def __init__(self):
        super(ContactEncoder,self).__init__()
        self.layers = Encoder()
    def forward(self,X):
        batch_size = X.size(0)
        if contact:
            blocks = X.unfold(1, m, stride).unfold(2, m, stride) if overlap else X.unfold(1, m, m).unfold(2, m, m)
            concatenated_blocks = blocks.reshape(batch_size, -1, m) if col else blocks.permute(0, 3, 1, 2, 4).reshape(batch_size, m, -1)
            u,u0 = self.layers(concatenated_blocks)
                
            u = u.unfold(1,m,m).unfold(2,m,m)

            if overlap:
                block_num = int((n - m) / stride + 1)
                u = u.reshape(batch_size,block_num,block_num,m,m)
                u = u.permute(0, 1, 3, 2, 4).reshape(batch_size, m*block_num, m*block_num)
                remove = torch.cat([torch.arange(m * i, m * i + (m-stride)) for i in range(1, block_num)])

                full = torch.tensor(range(m*block_num))
                selected_ind = full[torch.logical_not(torch.isin(full,remove))]

                u = torch.index_select(u, 1, selected_ind)
                u = torch.index_select(u, 2, selected_ind)
                #print(u.size())

            else:
                u = u.reshape(batch_size,int(n/m),int(n/m),m,m)
                u = u.permute(0, 1, 3, 2, 4).reshape(batch_size, 28, 28)
        else:
            X = X.view(batch_size,int(784/d),d)
            u,u0 = self.layers(X)
            u = u.view(batch_size,28,28)
        return u,u0



        




