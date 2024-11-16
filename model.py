import io
import torch
import torch.nn as nn
import numpy as np
import chess.pgn
from chess import Board
import torch.nn.functional as F

PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"

def encode_board(board: Board) -> np.array:
    step = 1 - 2 * board.turn
    unicode = board.unicode().replace(' ','').replace('\n','')[::step]
    return np.array([PIECE_CHARS[::step].index(c) for c in unicode], dtype=int).reshape(8,8)

class Attention(nn.Module):
    def __init__(self,input_dims,attention_dims,n_heads = 2):
        super().__init__()
        self.attention_dims = attention_dims
        self.n_heads = n_heads
        self.k1 = nn.Linear(input_dims, attention_dims)
        self.v1 = nn.Linear(input_dims, attention_dims)
        self.q1 = nn.Linear(input_dims, attention_dims)
        
        if n_heads == 2:
            self.k2 = nn.Linear(input_dims, attention_dims)
            self.v2 = nn.Linear(input_dims, attention_dims)
            self.q2 = nn.Linear(input_dims, attention_dims)
            self.attention_head_projection = nn.Linear(attention_dims * 2,input_dims)
        else:
            self.attention_head_projection = nn.Linear(attention_dims,input_dims)

        self.activation = nn.Softmax(dim = -1)
        
    def forward(self,x):
        oB, oD, oW, oH = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(oB, -1, oD)

        q1,v1,k1 = self.q1(x),self.v1(x),self.k1(x)
        qk1 = (q1@k1.permute((0,2,1)))/(self.attention_dims ** 0.5)
        multihead = self.activation(qk1)@v1 
        if self.n_heads == 2:
            q2,v2,k2 = self.q2(x),self.v2(x),self.k2(x)
            qk2 = (q2@k2.permute((0,2,1)))/(self.attention_dims ** 0.5) 
            attention = self.activation(qk2)@v2       
            multihead = torch.cat((multihead, attention),dim=-1)
   
        multihead_concat = self.attention_head_projection(multihead)
        return multihead_concat.reshape(oB, oD, oW, oH)

class Residual(nn.Module):
    def __init__(self, outer_channels, inner_channels, use_1x1conv, dropout, dilation = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(outer_channels, inner_channels, kernel_size=3, padding='same', stride=1, dilation = dilation)
        self.conv2 = nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding='same', stride=1, dilation = dilation)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(outer_channels, outer_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.bn2 = nn.BatchNorm2d(outer_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.dropout(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ChessModel(nn.Module):
    def __init__(self, nlayers, embed_dim, inner_dim, attention_dim, use_1x1conv, dropout, device='cpu'):
        super().__init__()
        self.vocab = PIECE_CHARS
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.use_1x1conv = use_1x1conv
        self.dropout = dropout

        # Shared layers
        self.embedder = nn.Embedding(len(self.vocab), self.embed_dim)
        self.convLayers = nn.ModuleList()
        for i in range(nlayers): 
            self.convLayers.append(Residual(self.embed_dim, self.inner_dim, self.use_1x1conv, self.dropout, 2**i))
            self.convLayers.append(Attention(self.embed_dim, attention_dim))

        self.convnet = nn.Sequential(*self.convLayers)
        self.accumulator = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=8, padding=0, stride=1)
        
        # Value head for evaluation
        self.value_head = nn.Linear(self.embed_dim, 1)
        
        # Policy head for RL
        self.policy_head = nn.Linear(self.embed_dim, 1968)  # Maximum possible moves in chess

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
        nn.init.uniform_(self.value_head.weight, -initrange, initrange)
        nn.init.uniform_(self.policy_head.weight, -initrange, initrange)

    def forward(self, inputs, return_policy=False):
        inputs = self.embedder(inputs)
        inputs = torch.permute(inputs, (0, 3, 1, 2)).contiguous() 
        features = self.convnet(inputs)
        features = F.relu(self.accumulator(features).squeeze())
        
        value = self.value_head(features).flatten()
        
        if return_policy:
            policy = F.softmax(self.policy_head(features), dim=-1)
            return value, policy
        return value

    def score(self, pgn, move):
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        for past_move in list(game.mainline_moves()):
            board.push(past_move)
        board.push_san(move)
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()