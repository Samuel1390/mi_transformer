import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Config:
    def __init__(self, vocab_size:int,
                 max_seq_length:int,
                 seq_length:int,
                 d_model:int,
                 num_layers:int,
                 num_heads:int,
                 d_ff:int,
                 device:str,
                 dropout:float):

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim:int = int(d_model / num_heads)

        self.vocab_size:int = vocab_size
        self.max_seq_length:int = max_seq_length
        self.seq_length:int = seq_length
        self.d_model:int = d_model
        self.num_layers:int = num_layers
        self.num_heads:int = num_heads
        self.d_ff:int = d_ff
        self.device:str = device
        self.dropout:float = dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.max_seq_length = config.max_seq_length
        self.seq_length = config.seq_length
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)
        self.register_buffer('mask', torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)).view(1,1,config.max_seq_length,config.max_seq_length))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        Q = self.W_q(x).view(-1, self.seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(x).view(-1, self.seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_v(x).view(-1, self.seq_length, self.num_heads, self.head_dim).transpose(1,2)

        attention = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(self.mask[:,:,:self.max_seq_length, :self.max_seq_length] == 0, float('-inf'))
        attention = self.dropout(F.softmax(attention, dim=-1))
        scores = torch.matmul(attention, V)
        scores = scores.transpose(1,2).contiguous().view(-1, self.seq_length, self.num_heads * self.head_dim)
        output = self.W_o(scores)
        return self.dropout(output)

class SwiGLUFFN(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff * 2, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, value = self.up_proj(x).chunk(2, dim=-1)
        swish_output = F.silu(gate) * value
        swish_output = self.dropout(swish_output)
        return self.down_proj(swish_output)

class Transformer(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.attention_layer = MultiHeadAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.d_model, config.d_model)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x) -> torch.Tensor:
        x = self.norm1(self.attention_layer(x)) + x
        x = self.norm2(self.ffn(x)) + x
        return x

class GPT2(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.device = config.device
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model).to(config.device)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.d_model).to(config.device)
        self.transformers = nn.Sequential(*[Transformer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.size()
        pos = torch.arange(0, seq_length).unsqueeze(0).to(self.device)
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.linear(x)
        return x
