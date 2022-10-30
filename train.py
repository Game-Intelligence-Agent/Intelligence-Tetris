import torch
torch.backends.cudnn.benchmark = True

from Games import Tetris
from Logger import Logger
from Models.Parameters import hyper_loader
from Models.Wrappers import *
from Models.utils import build_optimizer

import random
from copy import deepcopy
from Models.utils.tensorboard_handler import tb_handler
from tqdm import tqdm
from datetime import datetime
from statistics import mean, median
from collections import deque
import numpy as np


def get_args():
    import argparse
    parser = argparse.ArgumentParser("""Parameters for Tetris Agent""")
    parser.add_argument("--hyper_name", type = str, default = 'gnn_hyper', help = 'file name for hyper parameters')
    args = parser.parse_args()

    args.agent_hyper = hyper_loader('agent_hyper')
    args.train_hyper = hyper_loader(args.hyper_name)

    return args

# Run dqn with Tetris
def main():

    ## params
    args = get_args()
    log_every = args.train_hyper['train_params']['log_every']

    ## env
    env = Tetris(width = 10, height = 20, block_size = 30, mode = args.train_hyper['game_mode'])

    ## model
    model = eval(args.train_hyper['wrapper_name'])(**args.train_hyper['model_params']).to(args.train_hyper['train_params']['device'])
    target_model = deepcopy(model)
    name = args.train_hyper['model_params']['model_name'] + '_' + args.train_hyper['model_params']['model_type']
    tb = tb_handler('./Models/runs/', f'{name}', model)
    scheduler, optimizer = build_optimizer(args.train_hyper['optimizer_params'], tb.model.parameters())

    # print(tb.tb.model.state_dict())

    scores, tetrominoes, cleared_lines = [], [], []

    train_time, best_max, best_avg, best_min = 0, 0, 0, 100
    best_model = None

    replay_memory = deque(maxlen = args.agent_hyper['mem_size'])

    episode = 0
    print('start to play')
    state = env.reset()
    state = state.to(args.train_hyper['train_params']['device'])
    while episode < args.train_hyper['train_params']['episodes']:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = args.agent_hyper['epsilon_min'] + (max(args.agent_hyper['epsilon_stop_episode'] - episode, 0) * (
                args.agent_hyper['epsilon'] - args.agent_hyper['epsilon_min']) / args.agent_hyper['epsilon_stop_episode'])
        u = random.random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.to(args.train_hyper['train_params']['device'])
        tb.model.eval()
        with torch.no_grad():
            predictions = tb.model(next_states)[:, 0]
        tb.model.train()
        if random_action:
            index = random.randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        if args.train_hyper['train_params']['render_every'] is not None and episode > 0 and episode % args.train_hyper['train_params']['render_every'] == 0:
            render = True
        else:
            render = False
        reward, done = env.step(action, render = render)

        if torch.cuda.is_available():
            next_state = next_state.to(args.train_hyper['train_params']['device'])
        replay_memory.append([state, reward, next_state, done])

        if done:
            scores.append(env.score)
            tetrominoes.append(env.tetrominoes)
            cleared_lines.append(env.cleared_lines)
            state = env.reset()
            if torch.cuda.is_available():
                state = state.to(args.train_hyper['train_params']['device'])
        else:
            state = next_state
            continue

        if len(replay_memory) < args.agent_hyper['replay_start_size'] / 10:
            continue

        print(f'training for epoch {episode}')
        episode += 1
        ## 需要调整更新频率 看完再更新 每次预测不变
        target_model.load_state_dict(tb.model.state_dict())
        optimizer.zero_grad(set_to_none = True)
        for epoch in range(args.train_hyper['train_params']['epochs']):
            batch = random.sample(replay_memory, min(len(replay_memory), args.train_hyper['train_params']['batch_size']))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = torch.stack(tuple(state for state in state_batch))
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.stack(tuple(state for state in next_state_batch))

            if torch.cuda.is_available():
                state_batch = state_batch.to(args.train_hyper['train_params']['device'])
                reward_batch = reward_batch.to(args.train_hyper['train_params']['device'])
                next_state_batch = next_state_batch.to(args.train_hyper['train_params']['device'])

            target_model.eval()
            with torch.no_grad():
                next_prediction_batch = target_model(next_state_batch)

            tb.model.train()
            q_values = tb.model(state_batch)
                
            y_batch = torch.cat(
                tuple(reward if done else reward + args.agent_hyper['discount'] * prediction for reward, done, prediction in
                    zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

            loss = tb.model.loss(q_values, y_batch)
            loss.backward()
        optimizer.step()

        if log_every and episode and episode % log_every == 0:

            print(f'Episode {episode}----------------------------------------------')

            max_score, _, _ = Logger.log_result(tb, 'score', scores, log_every, episode)
            Logger.log_result(tb, 'cleared_line', cleared_lines, log_every, episode)
            Logger.log_result(tb, 'tetrominoe', tetrominoes, log_every, episode)

            if best_max <= max_score:
                best_max = max_score
                best_model = deepcopy(tb.model)

        if episode > 0 and episode % args.train_hyper['train_params']['save_every'] == 0:
            # print(best_tb.model.state_dict())
            best_model.save_parameters()

if __name__ == "__main__":
    main()
