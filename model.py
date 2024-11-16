import io
import torch
import torch.nn as nn
import numpy as np
import chess.pgn
from chess import Board
import torch.nn.functional as F

PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"

def encode_board(board: Board) -> np.array:
    # String-encode the board.
    # If board.turn = 1 then it is now white's turn which means this is a potential move
    # being contemplated by black, and therefore we reverse the char order to rotate the board
    # for black's perspective
    # If board.turn = 0 then it is now black's turn which means this is a potential move
    # being contemplated by white, and therefore we leave the char order as white's perspective.
    # Also reverse PIECE_CHARS indexing order if black's turn to reflect "my" and "opponent" pieces.
    step = 1 - 2 * board.turn
    unicode = board.unicode().replace(' ','').replace('\n','')[::step]
    return np.array([PIECE_CHARS[::step].index(c) for c in unicode], dtype=int).reshape(8,8)

class Residual(nn.Module):
    """
    The Residual block of ResNet models.
    """
    def __init__(self, outer_channels, inner_channels, use_1x1conv, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(outer_channels, inner_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(outer_channels, outer_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.bn2 = nn.BatchNorm2d(outer_channels)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if self.conv3 is not None:
            nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
            if self.conv3.bias is not None:
                nn.init.constant_(self.conv3.bias, 0)
        
        for module in [self.bn1, self.bn2]:
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.dropout(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Model(nn.Module):
    def __init__(self, nlayers, embed_dim, inner_dim, use_1x1conv, dropout, device='cpu'):
        super().__init__()
        self.vocab = PIECE_CHARS
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.use_1x1conv = use_1x1conv
        self.dropout = dropout

        # Initial convolutional layer (similar to AlphaZero)
        self.embedder = nn.Embedding(len(self.vocab), self.embed_dim)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Residual tower (similar to AlphaZero's 19-20 residual blocks)
        self.convnet = nn.Sequential(*[
            Residual(256, 256, self.use_1x1conv, self.dropout) 
            for _ in range(nlayers)
        ])

        # Policy head (simplified since we only need value output)
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs):
        x = self.embedder(inputs)
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        x = self.initial_conv(x)
        x = self.convnet(x)
        scores = self.value_head(x).flatten()
        return scores

    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # init a game and board
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        # catch board up on game to present
        for past_move in list(game.mainline_moves()):
            board.push(past_move)
        # push the move to score
        board.push_san(move)
        # convert to tensor, unsqueezing a dummy batch dimension
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()