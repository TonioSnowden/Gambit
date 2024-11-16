import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import argparse
import os
import chess
import numpy as np
from cycling_utils import TimestampedTimer, AtomicDirectory
from model import ChessModel, encode_board

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default="/root/chess-hackathon-4/model_config.yaml")
    parser.add_argument("--save-dir", type=Path, default=os.environ["OUTPUT_PATH"])
    parser.add_argument("--load-path", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-episodes", type=int, default=1000)
    return parser

def get_legal_moves_mask(board):
    mask = torch.zeros(1968)  # Max possible moves
    for move in board.legal_moves:
        mask[move.from_square * 64 + move.to_square] = 1
    return mask

def play_game(model, device):
    board = chess.Board()
    moves = []
    rewards = []
    
    while not board.is_game_over():
        state = torch.tensor(encode_board(board)).unsqueeze(0).to(device)
        value, policy = model(state, return_policy=True)
        
        # Mask illegal moves
        legal_moves_mask = get_legal_moves_mask(board).to(device)
        masked_policy = policy * legal_moves_mask
        
        # Sample move from policy
        move_idx = torch.multinomial(masked_policy, 1).item()
        from_square = move_idx // 64
        to_square = move_idx % 64
        move = chess.Move(from_square, to_square)
        
        moves.append((state, move_idx))
        board.push(move)
        
        # Simple reward: +1 for checkmate, -1 for getting checkmated, 0 otherwise
        if board.is_checkmate():
            reward = 1 if board.turn else -1
        else:
            reward = 0
        rewards.append(reward)
    
    return moves, rewards

def main(args):
    dist.init_process_group("nccl")
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)

    # Load pre-trained model
    model = ChessModel(**model_config).to(device_id)
    checkpoint = torch.load(args.load_path, map_location=f"cuda:{device_id}")
    model.load_state_dict(checkpoint["model"])
    model = DDP(model, device_ids=[device_id])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for episode in range(args.num_episodes):
        moves, rewards = play_game(model.module, device_id)
        
        # Calculate returns (discounted rewards)
        returns = []
        R = 0
        gamma = 0.99
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device_id)
        
        # Update policy
        policy_loss = 0
        value_loss = 0
        for (state, move_idx), R in zip(moves, returns):
            value, policy = model(state, return_policy=True)
            policy_loss -= torch.log(policy[0, move_idx]) * R
            value_loss += (value - R) ** 2
        
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0 and dist.get_rank() == 0:
            print(f"Episode {episode}, Loss: {loss.item():.3f}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)